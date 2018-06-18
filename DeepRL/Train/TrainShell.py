import cmd


class TrainShell(cmd.Cmd):
    intro = '[[ Welcome to the shell.   Type help or ? to list commands. ]]'
    prompt = '>'

    def __init__(self, _trainer):
        super().__init__()
        self.trainer = _trainer

    def do_save(self, _arg):
        self.trainer.agent.save(
            self.trainer.epoch, self.trainer.step_local
        )

    def do_eval(self, _arg):
        self.trainer.agent.evaluating()
        self.trainer.agent.startNewGame()
        while self.trainer.agent.step():
            pass
        self.trainer.agent.training()

    def do_quit(self, _arg):
        return True
