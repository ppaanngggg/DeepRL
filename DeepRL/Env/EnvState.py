import typing


class EnvState(object):
    def __init__(
            self, _in_game: bool,
            _dict: typing.Dict
    ):
        assert isinstance(_dict, dict)

        self.in_game = _in_game
        self.state = _dict

    def __repr__(self) -> str:
        tmp = \
            "##### State #####\n" \
            "## in_game: {}\n" \
            "## states: {}\n" \
            "#################"
        return tmp.format(
            self.in_game, self.state,
        )
