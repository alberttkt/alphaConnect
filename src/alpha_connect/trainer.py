import os
import random
import numpy as np
from random import shuffle
import torch
import torch.optim as optim
from tqdm import tqdm
from copy import deepcopy

from .alpha_zero_agent import AlphaZeroAgent
from .neural_net_agent import NeuralNetAgent
from .helper import state_to_supervised_input
from .game_choice import GameChoice

from game import ConnectState


class Trainer:
    def __init__(self, model, args):
        self.num_players = GameChoice.get_number_players()
        self.model = model
        self.args = args
        self.agents = [
            AlphaZeroAgent(NeuralNetAgent(model), self.args["num_simulations"])
            for _ in range(self.num_players)
        ]

    def exceute_episode(self):
        train_examples = []
        state = ConnectState.sample_initial_state()

        agents = self.agents.copy()
        random.shuffle(agents)

        i = 0

        while True:
            current_player = state.player

            action_probs, _ = agents[current_player].play(state)

            network_input = state_to_supervised_input(state)
            action_probs_list = [action_probs.get(i, 0) for i in range(7)]
            train_examples.append((network_input, current_player, action_probs_list))

            action = random.choices(
                list(action_probs.keys()), weights=action_probs.values()
            )[0]

            state = action.sample_next_state()
            reward = state.reward

            i += 1

            if state.has_ended:
                ret = []
                for (
                    hist_state,
                    hist_current_player,
                    hist_action_probs,
                ) in train_examples:
                    # [Board, actionProbabilities, Reward]
                    ret.append(
                        (hist_state, hist_action_probs, reward[hist_current_player])
                    )

                print(f"Game ended after {i} steps")
                return ret

    def learn(self):
        for i in range(1, self.args["numIters"] + 1):
            print("{}/{}".format(i, self.args["numIters"]))

            train_examples = []

            for eps in tqdm(range(self.args["numEps"])):
                iteration_train_examples = self.exceute_episode()
                train_examples.extend(iteration_train_examples)

            shuffle(train_examples)
            self.train(train_examples)
            filename = self.args["checkpoint_path"]
            self.save_checkpoint(folder=".", filename=filename)
            self.agents = self.agents[1::]
            self.agents.append(
                AlphaZeroAgent(
                    NeuralNetAgent(self.new_model), self.args["num_simulations"]
                )
            )

    def train(self, examples):
        self.model = deepcopy(self.model)
        optimizer = optim.Adam(self.model.parameters(), lr=5e-4)
        pi_losses = []
        v_losses = []

        for epoch in range(self.args["epochs"]):
            self.model.train()

            batch_idx = 0

            while batch_idx < int(len(examples) / self.args["batch_size"]):
                sample_ids = np.random.randint(
                    len(examples), size=self.args["batch_size"]
                )
                boards, pis, vs = list(zip(*[examples[i] for i in sample_ids]))
                boards = torch.FloatTensor(np.array(boards).astype(np.float64))
                # print(pis)
                target_pis = torch.FloatTensor(np.array(pis))
                target_vs = torch.FloatTensor(np.array(vs).astype(np.float64))

                # predict

                # boards = boards.contiguous().cuda()
                # target_pis = target_pis.contiguous().cuda()
                # target_vs = target_vs.contiguous().cuda()

                # compute output
                model_input = boards.reshape(-1, 3, 6, 7).to("mps")
                out_pi, out_v = self.model(model_input)
                out_pi, out_v = out_pi.to("cpu"), out_v.to("cpu")
                l_pi = self.loss_pi(target_pis, out_pi)
                l_v = self.loss_v(target_vs, out_v)
                total_loss = l_pi + l_v

                pi_losses.append(float(l_pi))
                v_losses.append(float(l_v))

                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()

                batch_idx += 1

            print()
            print("Policy Loss", np.mean(pi_losses))
            print("Value Loss", np.mean(v_losses))
            print("Examples:")
            print(out_pi[0].detach())
            print(target_pis[0])

    def loss_pi(self, targets, outputs):
        # remove negative values from the output
        outputs = torch.clamp(outputs, min=1e-8, max=1.0)
        loss = -(targets * torch.log(outputs)).sum(dim=1)
        return loss.mean()

    def loss_v(self, targets, outputs):
        loss = torch.sum((targets - outputs.view(-1)) ** 2) / targets.size()[0]
        return loss

    def save_checkpoint(self, folder, filename):
        if not os.path.exists(folder):
            os.mkdir(folder)

        filepath = os.path.join(folder, filename)
        torch.save(
            {
                "state_dict": self.new_model.state_dict(),
            },
            filepath,
        )
