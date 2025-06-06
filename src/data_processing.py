def get_categories():
    """Возвращает существующие на сайте категории и их индустрии.

    Returns:
        list: список индустрий и их категорий."""

    url = "https://api.hh.ru/industries"
    host = "hh.ru"
    locale = "RU"
    res = req.get(url, params={
        "host": host,
        "locale": locale
    })
    return res

def vacs_by_industry_id(id, page=0):
    """Возвращает страницу с вакансиями по индустрии.

    Args: 
        id (float): id индустрии.
        page (int): номер страницы с вакансиями.

    Returns:
        list: список вакансий."""

    res = req.get("https://api.hh.ru/vacancies", params={
        "industry": id,
        "per_page": 100,
        "page": page,
        "locale": "RU",
        "host": "hh.ru",
        "only_with_salary": "true",
    })
    return res

def vacancy_verifier(vacancy):
    """Проверяет, соответсвует ли вакансия требованиям.
    
    Args:
        vacancy (list): необработанная вакансия.
    
    Returns:
        boolean: соответствует ли вакансия заданным требованиям.
    
    Note:
        Передаваемая вакансия должна содержать все стандартные параметры."""

    if not (vacancy["area"]["name"] in city.keys()):
        return False
    if not vacancy["work_format"]:
        return False
    if not (vacancy["salary"]["currency"] in ["USD", "RUR", "EUR"]):
        return False
    if not vacancy["approved"]:
        return False
    if not vacancy["salary"]["from"]:
        return False

    year, month, day = vacancy['published_at'][:10].split('-')
    today = datetime.date.today()
    vacancy_date = datetime.date(int(year), int(month), int(day))
    delta = abs(today - vacancy_date)
    if delta.days > 80:
        return False

    return True
