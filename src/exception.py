import sys
import logger

def error_message_detail(error, error_detail:sys):
    """When an error occurs, this function prints the error message"""
    _, _, exc_tb = error_detail.exc_info() # exc_info gives you information on which line and file the error occurred
    file_name = exc_tb.tb_frame.f_code.co_filename
    line_number = exc_tb.tb_lineno
    error_message = f"Error in python script in line {line_number} of file {file_name}: {str(error)}"
    
    return error_message
    
class CustomException(Exception):
    def __init__(self, error_message, error_detail:sys):
        super().__init__(error_message) # We inherit the error message from the Exception class
        self.error_message = error_message_detail(error_message, error_detail = error_detail)
    
    def __str__(self): # When we print the object, we print the error message
        return self.error_message