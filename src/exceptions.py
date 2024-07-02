import sys
#import logger
import logging

def error_message_details(error, error_details:sys):

    ## Find out the file and the line number in which exception raised
    _, _, exc_tb = error_details.exc_info()

    if exc_tb is None:
        return f"Error: {str(error)} No traceback available..."

    ## Extract the filename, linenumber and error message

    filename = exc_tb.tb_frame.f_code
    linenumber = exc_tb.tb_lineno

    error_message = f"Error: {str(error)} Found in {filename}, Line Number: {linenumber}"

    return error_message

class Custom_exception(Exception):
    def __init__(self, error_message, error_details:sys):
        super().__init__(error_message)
        self.error_message = error_message_details(error_message, 
                                                   error_details = error_details)
    
    def __str__(self):
        return self.error_message
    



