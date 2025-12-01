from datetime import datetime

def get_user_date():
    date_str = input("Enter a date (DD-MM-YYYY): ")
    user_date = datetime.strptime(date_str, "%d-%m-%Y")
    return user_date


def check_date(user_date):
    today = datetime.now().date()
    if user_date.date() < today:
        print(f"The date {user_date.strftime('%d-%m-%Y')} is in the past.")
    elif user_date.date() == today:
        print(f"The date {user_date.strftime('%d-%m-%Y')} is today.")
    else:
        print(f"The date {user_date.strftime('%d-%m-%Y')} is in the future.")


def main():
    user_date = get_user_date()
    if user_date:
        check_date(user_date)


main()
