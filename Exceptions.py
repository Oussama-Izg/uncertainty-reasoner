class ConstraintException(Exception):
    """
    To throw, if a constraint is not met.
    """
    def __init__(self, message):
        super().__init__(message)
