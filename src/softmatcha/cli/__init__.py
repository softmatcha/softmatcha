import termcolor


def highlight(string: str) -> str:
    return termcolor.colored(string, "red")
