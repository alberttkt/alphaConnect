from game import Connect4State


class GameChoice:
    game = Connect4State

    @classmethod
    def set_game(cls, new_game):
        cls.game = new_game

    @classmethod
    def get_game(cls):
        return cls.game

    @classmethod
    def get_number_players(cls):
        return len(cls.game.sample_initial_state().reward)
