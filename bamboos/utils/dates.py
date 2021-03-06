import datetime

DAYS_IN_MONTH = 30.41
DAYS_IN_YEAR = 365.25
MONTH_IN_YEAR = 12.
SECOND_IN_MINUTE = 60.
SECOND_IN_HOUR = 3600.
MINUTE_IN_HOUR = 60.
HOUR_IN_DAY = 24.
BUSINESS_OPEN = 9.
BUSINESS_CLOSE = 17.
MIDNIGHT_START = -1
MORNING_START = 6
AFTERNOON_START = 12
NIGHT_START = 18
NIGHT_END = 24
SATURDAY = 5
SUNDAY = 6


def get_datetime():
    """
    Returns the current Date and Time in the following format : "%Y%m%d-%H%M%S"
    Returns:
        (str) : current Date and Time in the following format : "%Y%m%d-%H%M%S"
    """
    return datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
