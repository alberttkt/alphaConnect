import os
import random
import numpy as np
from random import shuffle
import torch
import torch.optim as optim
from tqdm import tqdm
from copy import deepcopy

from .alpha_zero_agent import AlphaZeroAgent, AlphaZeroTrainer
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

    def execute_episode(self):
        train_examples = []
        state = GameChoice.get_game().sample_initial_state()

        agents = self.agents.copy()
        random.shuffle(agents)

        i = 0

        while True:
            current_player = state.player

            action_probs, _ = agents[current_player].play(state)

            network_input = state_to_supervised_input(state)
            int_action_probs = {k.to_json(): v for k, v in action_probs.items()}
            action_probs_list = [int_action_probs.get(i, 0) for i in range(7)]
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

                # print(f"Game ended after {i} steps")
                return ret, self.agents.index(agents[current_player])

    def learn(self):
        best_loss = float("inf")
        best_model = None
        patience = 0
        for i in range(1, self.args["numIters"] + 1):
            print("{}/{}".format(i, self.args["numIters"]))

            train_examples = []
            winner_dict = {}
            alphaZeroTrainer = AlphaZeroTrainer(
                self.model,
                nb_episodes=self.args["numEps"],
                iterations=self.args["num_simulations"],
                temperature=self.args["temperature"],
            )

            results = alphaZeroTrainer.execute_episodes()
            train_examples.extend(results)

            shuffle(train_examples)
            loss = self.train(train_examples)
            if loss < best_loss:
                print(f"New best loss: {loss}")
                best_loss = loss
                best_model = deepcopy(self.model)
            else:
                patience += 1
                if patience >= 100:
                    print(f"Early stopping at iteration {i}")
                    break
            with open(self.args["loss_history_path"], "a") as f:
                f.write(str(loss))

            filename = self.args["checkpoint_path"]
            self.save_checkpoint(folder=".", filename=filename)

            self.agents = self.agents[1::]
            self.agents.append(
                AlphaZeroAgent(NeuralNetAgent(self.model), self.args["num_simulations"])
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

                target_pis = torch.FloatTensor(np.array(pis))
                target_vs = torch.FloatTensor(np.array(vs).astype(np.float64))

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
            print(f"loss: {3*np.mean(pi_losses) + np.mean(v_losses)}")
            print(f"Policy Loss", np.mean(pi_losses))
            print("Value Loss", np.mean(v_losses))
            print("Examples:")
            print(out_pi[0].detach())
            print(target_pis[0])
            print(out_v[0].detach())
            print(target_vs[0])

            return 3 * np.mean(pi_losses) + np.mean(v_losses)

    def loss_pi(self, targets, outputs):
        # mean squared error
        return torch.sum((targets - outputs) ** 2) / targets.size()[0]

    def loss_v(self, targets, outputs):
        loss = torch.sum((targets - outputs.view(-1)) ** 2) / targets.size()[0]
        return loss

    def save_checkpoint(self, folder, filename):
        if not os.path.exists(folder):
            os.mkdir(folder)

        filepath = os.path.join(folder, filename)
        torch.save(self.model.state_dict(), filepath)
