import sys
from src.logger import logging

class CustomException(Exception):
    def __init__(self, error_message, error_detail: sys):
        super().__init__(error_message)
        self.error_message = self.get_detailed_error_message(str(error_message), error_detail)

    def get_detailed_error_message(self, error_message, error_detail: sys):
        """
        Constructs a detailed error message including the original error message and traceback.
        """
        _, _, exc_tb = error_detail.exc_info()  # Get exception details
        file_name = exc_tb.tb_frame.f_code.co_filename  # Get the filename where the exception occurred
        line_number = exc_tb.tb_lineno  # Get the line number of the error
        
        detailed_message = f"Error occurred in file [{file_name}] at line number [{line_number}] with message [{error_message}]"
        return detailed_message

    def __str__(self):
        return self.error_message


# Example usage of the CustomException
if __name__ == "__main__":
    try:
        a = 1 / 0  # This will raise a ZeroDivisionError
    except Exception as e:
        logging.info("Divide by Zero")
        raise CustomException(str(e), sys)
