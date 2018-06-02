import typing


class EnvState(object):
    """
    This is the env state tuple, which will be stored in replay buffer

    _in_game: whether it is in game
    _state: env data
    """

    def __init__(
            self, _in_game: bool,
            _state: typing.Any
    ):
        self.in_game = _in_game
        self.state = _state

    def __repr__(self) -> str:
        tmp = \
            "## EnvState ##\n" \
            "  in_game: {}\n" \
            "  states: {}"
        return tmp.format(
            self.in_game, self.state,
        )
