import sys


def error_message_detail(error: Exception, error_detail: sys) -> str:
    """
    Extracts filename, line number and error message from the exception traceback.

    :param error: The exception that occured.
    :param error_detail: The sys module to access traceback details.
    :return: A formatted error message string.
    """
    _, _, exc_tb = error_detail.exc_info()
    file_name = exc_tb.tb_frame.f_code.co_filename
    line_number = exc_tb.tb_lineno

    return (
        f"Error occurred in script: [{file_name}] "
        f"at line: [{line_number}] "
        f"with message: [{str(error)}]"
    )


class WeatherException(Exception):
    def __init__(self, error_message: Exception, error_detail: sys):
        super().__init__(str(error_message))
        self.error_message = error_message_detail(error_message, error_detail)

    def __str__(self) -> str:
        return self.error_message