class ProcessingFailure(RuntimeError):
    def __init__(self, code, message, retryable=False):
        self.code, self.message, self.retryable = code, message, retryable
        super().__init__(message)


class LeaseLost(RuntimeError):
    pass


class WorkerStopped(RuntimeError):
    pass


class SubmissionUncertain(RuntimeError):
    """A write may have succeeded; abandon the lease rather than replay the write."""
