def initialize_encoding_maps():
    """Создаёт словари для кодирования категориальных признаков.

    Returns:
        dict: Словарь, где ключи - это названия признаков,
              а значения - это словари мэппинга.

    Note:
        Созданные словари необходимы для работы функции vacancy_refiner"""

    features = ["accept_handicapped", 
    "accept_kids", 
    "accept_temporary", 
    "area", 
    "driver_license_types", 
    "employment_form", 
    "experience", 
    "internship",
    "key_skills",
    "languages", 
    "night_shifts", 
    "salary", 
    "work_format", 
    "work_schedule_by_days", 
    "working_hours" 
     ]
    
    features_dict = req.get(f"https://api.hh.ru/dictionaries").json()
    
    employment_form = {}
    i = 0
    for item in features_dict["employment_form"]:
        employment_form[item["id"]] = i
        i += 1
    
    experience = {}
    i = 0
    for item in features_dict["experience"]:
        experience[item["id"]] = i
        i += 1
    
    work_format = {}
    i = 0
    for item in features_dict["work_format"]:
        work_format[item["id"]] = i
        i += 1
    
    work_schedule_by_days = {}
    i = 0
    for item in features_dict["work_schedule_by_days"]:
        work_schedule_by_days[item["id"]] = i
        i += 1
    
    working_hours = {}
    i = 0
    for item in features_dict["working_hours"]:
        working_hours[item["id"]] = i
        i += 1
    
    city = {
        "Москва": 0,
        "Санкт-Петербург": 1,
        "Новосибирск": 2,
        "Екатеринбург": 3,
        "Казань": 4,
        "Нижний Новгород": 5,
        "Красноярск": 6,
        "Челябинск": 7,
        "Самара": 8,
        "Уфа": 9,
        "Ростов-на-Дону": 10,
        "Краснодар": 11,
        "Омск": 12,
        "Воронеж": 13,
        "Пермь": 14,
        "Волгоград": 15,
        "Тюмень": 16,
        "Саратов": 17,
        "Тольятти": 18,
        "Барнаул": 19,
    }

def vacancy_refiner(vacancy):
    """Обрабатывае вакансию для использования в обучении.

    Args:
        vacancy (list): необработанная вакансия.

    Returns:
        list: обработанная вакансия.

    Note:
        Для работы функции необходимо использовать функцию initialize_encoding_maps"""

    refined_vacancy = {feature: (vacancy[feature] if vacancy[feature] != [] else None) for feature in features}

    refined_vacancy["accept_handicapped"] = 1 if refined_vacancy["accept_handicapped"] else 0
    refined_vacancy["accept_kids"] = 1 if refined_vacancy["accept_kids"] else 0
    refined_vacancy["accept_temporary"] = 1 if refined_vacancy["accept_temporary"] else 0
    refined_vacancy["internship"] = 1 if refined_vacancy["internship"] else 0
    refined_vacancy["night_shifts"] = 1 if refined_vacancy["night_shifts"] else 0
    refined_vacancy["driver_license"] = 1 if refined_vacancy["driver_license_types"] else 0
    refined_vacancy.pop("driver_license_types")

    refined_vacancy["area"] = refined_vacancy["area"]["name"]
    refined_vacancy["employment_form"] = refined_vacancy["employment_form"]["id"]

    if refined_vacancy["experience"]: refined_vacancy["experience"] = refined_vacancy["experience"]["id"]

    if refined_vacancy["key_skills"]: refined_vacancy["key_skills"] = [item["name"] for item in
                                                                       refined_vacancy["key_skills"]]
    if not refined_vacancy["key_skills"]: refined_vacancy["key_skills"] = []

    if refined_vacancy["languages"]: refined_vacancy["languages"], refined_vacancy["levels"] = \
        [item["id"] for item in refined_vacancy["languages"]], [item["level"]["id"] for item in
                                                                refined_vacancy["languages"]]

    if not refined_vacancy["languages"]: refined_vacancy["levels"] = None

    if refined_vacancy["languages"]:
        for i in range(len(refined_vacancy["languages"])):
            refined_vacancy["key_skills"].append(f'{refined_vacancy["languages"][i]}-{refined_vacancy["levels"][i]}')
    refined_vacancy.pop("languages")
    refined_vacancy.pop("levels")

    if refined_vacancy["work_format"]: refined_vacancy["work_format"] = refined_vacancy["work_format"][0]["id"]

    if refined_vacancy["work_schedule_by_days"]: refined_vacancy["work_schedule_by_days"] = \
    refined_vacancy["work_schedule_by_days"][0]["id"]

    if refined_vacancy["working_hours"]: refined_vacancy["working_hours"] = refined_vacancy["working_hours"][0]["id"]

    refined_vacancy["salary"], refined_vacancy["currency"] = \
        refined_vacancy["salary"]["from"], refined_vacancy["salary"]["currency"]

    refined_vacancy["employment_form"] = employment_form[refined_vacancy["employment_form"]]
    refined_vacancy["experience"] = experience[refined_vacancy["experience"]]
    refined_vacancy["work_format"] = work_format[refined_vacancy["work_format"]]
    refined_vacancy["work_schedule_by_days"] = work_schedule_by_days[refined_vacancy["work_schedule_by_days"]]
    refined_vacancy["working_hours"] = working_hours[refined_vacancy["working_hours"]]
    refined_vacancy["area"] = city[refined_vacancy["area"]]

    refined_vacancy["RUB_salary"] = refined_vacancy["salary"] * (1 if refined_vacancy["currency"] == "RUR" else 88)
    refined_vacancy.pop("currency")
    refined_vacancy.pop("salary")

    return refined_vacancy

def vacancies_refiner(vacancies):
    """Возвращает датафрейм pandas из переданных вакансий.

    Args:
        vacancies (list): список обработанных вакансий.

    Returns:
        pd.DataFrame: датафрейм pandas из переданных вакансий"""

    return pd.DataFrame(vacancies)

def skill_transform(vacancies):
    """Кодирует столбец 'key_skills' с использованием MultiLabelBinarizer
    и добавляет полученные бинарные признаки к исходному датафрейму.

    Args:
        vacancies_df (pd.DataFrame): фатафрейм с вакансиями.

    Returns:
        pd.DataFrame: DataFrame с удаленным исходным столбцом 'key_skills'
                      и добавленными новыми бинарными столбцами для каждого навыка.

    Note: Исходный датафрейм должен содержать столбец 'key_skills'."""

    skill_binarizer = MultiLabelBinarizer()
    encoded_skills = skill_binarizer.fit_transform(vacancies["key_skills"])
    encoded_skills_df = pd.DataFrame(encoded_skills, columns=skill_binarizer.classes_, index=vacancies.index)
    skill_encoded_vacancies = pd.concat([vacancies.drop('key_skills', axis=1), encoded_skills_df], axis=1)
    return skill_encoded_vacancies
