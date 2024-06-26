def printBoard(board):
    """
    Prints the game board as text output to the terminal.

    Parameters
    ----------
    board : list of lists
        the current game board
    """
    print("    0   1   2\n")
    for i, row in enumerate(board):
        print("%i   " % i, end="")
        for elt in row:
            print("%s   " % elt, end="")
        print("\n")


def getInput(message):
    """
    Print the message and return recived value
    """

    print("\n")
    return input(message)


def displayMessage(message):
    print(f"\n{message}")



