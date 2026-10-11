"""
Custom domain exceptions for CrediLens
"""

class CrediLensError(Exception):
    """Base exception for all CrediLens errors."""
    pass

class ModelNotLoadedError(CrediLensError):
    """Raised when an operation requires a model that is not yet loaded."""
    pass

class PredictionError(CrediLensError):
    """Raised when an error occurs during model prediction inference."""
    pass

class InvalidApplicantDataError(CrediLensError):
    """Raised when applicant features or CSV input fails validation."""
    pass
