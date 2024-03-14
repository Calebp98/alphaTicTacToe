import copy

class EpisodeGameTracker:
    def __init__(self, game):
        self.game = game
        self.currentGame = []  # Tracks board states for the current game
        self.boardLog = []  # Logs of all games played
        self.failures = []  # Logs of all games that are considered failures

    def setGame(self, game):
        """Set or reset the game to track."""
        self.game = game
    
    def logBoardState(self):
        """Log the current state of the game board."""
        self.currentGame.append(copy.deepcopy(self.game.board))
        
    def logGame(self, success=True):
        """Log the completed game and reset current game log."""
        self.boardLog.append(self.currentGame)
        if not success:
            # If the game is considered a failure, log it separately
            self.failures.append(self.currentGame)
        self.currentGame = []

    def getBoardLog(self):
        """Return the logged history of all games."""
        return self.boardLog

    def getFailures(self):
        """Return the logged history of all failed games."""
        return self.failures
