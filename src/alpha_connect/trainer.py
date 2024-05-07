import os
import random
import numpy as np
from random import shuffle
import torch
import torch.optim as optim
from tqdm import tqdm
from copy import deepcopy
import concurrent.futures

from .alpha_zero_agent import AlphaZeroAgent, AlphaZeroTrainer
from .neural_net_agent import NeuralNetAgent
from .helper import state_to_supervised_input
from .game_choice import GameChoice

from game import Connect4State


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
        start_index = 1
        open(self.args["loss_history_path"], "a").close()
        loss_file = open(self.args["loss_history_path"], "r")
        loss_lines = loss_file.readlines()
        if len(loss_lines) > 0:
            start_index = len(loss_lines)
            best_loss = min([float(x) for x in loss_lines])
            patience = len(loss_lines) - loss_lines.index(str(best_loss) + "\n")
            print(
                f"Resuming training from iteration {start_index} with best loss {best_loss}"
            )
        loss_file.close()
        lr = self.args["lr"]
        lr_decay = self.args["lr_decay"]

        for i in range(start_index, self.args["numIters"] + 1):
            lr *= lr_decay
            print("{}/{} with lr {}".format(i, self.args["numIters"], lr))

            train_examples = []
            alphaZeroTrainer = AlphaZeroTrainer(
                self.model,
                nb_episodes=self.args["numEps"],
                iterations=self.args["num_simulations"],
                temperature=self.args["temperature"],
            )
            with concurrent.futures.ThreadPoolExecutor() as executor:
                progress = tqdm(range(self.args["numEps"] * self.args["numThreads"]))
                futures = [
                    executor.submit(alphaZeroTrainer.execute_episodes, progress)
                    for _ in range(self.args["numThreads"])
                ]
                for future in concurrent.futures.as_completed(futures):
                    results = future.result()
                    train_examples.extend(results)
            progress.close()

            # results = alphaZeroTrainer.execute_episodes()
            # train_examples.extend(results)

            shuffle(train_examples)
            loss = self.train(train_examples, lr)
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
                f.write(str(loss) + "\n")

            filename = self.args["checkpoint_path"]
            self.save_checkpoint(folder=".", filename=filename)

            self.agents = self.agents[1::]
            self.agents.append(
                AlphaZeroAgent(NeuralNetAgent(self.model), self.args["num_simulations"])
            )

    def train(self, examples, lr):
        self.model = deepcopy(self.model)
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
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
            print(f"Loss: {3*np.mean(pi_losses) + np.mean(v_losses)}")
            print(f"Policy Loss", 3 * np.mean(pi_losses))
            print("Value Loss", np.mean(v_losses))
            print(f"{batch_idx} batches processed")
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
