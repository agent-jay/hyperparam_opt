import logging

def create(filename, log_name):
	logging.basicConfig(filename=filename,level=logging.DEBUG, disable_existing_loggers=False)
	# create logger
	logger = logging.getLogger(log_name)

	# create file handler 
	ch = logging.FileHandler(filename)

	# create formatter
	formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

	# add formatter to ch
	ch.setFormatter(formatter)

	# add ch to logger
	logger.addHandler(ch)

	#Set the level for the logger
	logger.setLevel(logging.DEBUG)
	
	logger.propagate = False        
	
	return logger

