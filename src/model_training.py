def train_linear_regression(X_train, y_train):
    """Обучает модель линейной регрессии.
    
    Args:
        X_train (pd.DataFrame): Данные нецелевых перменных.
        y_train (pd.DataFrame): Данные целевой переменной.
    
    Returns:
        model: Обученная модель линейной регрессии."""
    
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model

def train_lasso_regression(X_train, y_train):
    """Обучает модель линейной регрессии с L1-регуляризацией.
    
    Args:
        X_train (pd.DataFrame): Данные нецелевых перменных.
        y_train (pd.DataFrame): Данные целевой переменной.
    
    Returns:
        model: Обученная модель линейной регрессии с L1-регуляризацией."""
        
    model = Lasso()
    model.fit(X_train, y_train)
    return model

def train_random_forest(X_train, y_train):
    """Обучает модель случайного леса.
    
    Args:
        X_train (pd.DataFrame): Данные нецелевых перменных.
        y_train (pd.DataFrame): Данные целевой переменной.
    
    Returns:
        model: Обученная модель случайного леса.
        
    Note:
        Гиперпараметры подобраны специфично для проекта."""
        
    model = RandomForestRegressor(n_estimators=100, max_depth=500,
    oob_score=True, criterion='absolute_error', 
    max_features=0.05, random_state=69)
    model.fit(X_train, y_train)
    return model

def train_gradient_boosting(X_train, y_train):
    """Обучает модель градиентного бустинга.
    
    Args:
        X_train (pd.DataFrame): Данные нецелевых перменных.
        y_train (pd.DataFrame): Данные целевой переменной.
    
    Returns:
        model: Обученная модель градиентного бустинга.
        
    Note:
        Гиперпараметры подобраны специфично для проекта."""
        
    model = GradientBoostingRegressor(n_estimators=1500, loss="absolute_error", learning_rate=0.01)
    model.fit(X_train, y_train)
    return model

def train_ada_boosting(X_train, y_train):
    """Обучает модель адаптивного бустинга.
    
    Args:
        X_train (pd.DataFrame): Данные нецелевых перменных.
        y_train (pd.DataFrame): Данные целевой переменной.
    
    Returns:
        model: Обученная модель адаптивного бустинга.
        
    Note:
        Гиперпараметры подобраны специфично для проекта."""
        
    basic_LR = LinearRegression()
    model = AdaBoostRegressor(estimator=basic_LR, n_estimators=100)
    model.fit(X_train, y_train)
    return model

def train_xgboosting(X_train, y_train):
    """Обучает модель XG бустинга.
    
    Args:
        X_train (pd.DataFrame): Данные нецелевых перменных.
        y_train (pd.DataFrame): Данные целевой переменной.
    
    Returns:
        model: Обученная модель XG бустинга.
        
    Note:
        Гиперпараметры подобраны специфично для проекта."""
        
    model = xgb.XGBRegressor(n_estimators=750, grow_policy="lossguide", learning_rate=0.1)
    model.fit(X_train, y_train)
    return model

def train_mlp_regressor(X_train, y_train):
    """Обучает модель полносвязной нейронной сети.
    
    Args:
        X_train (pd.DataFrame): Данные нецелевых перменных.
        y_train (pd.DataFrame): Данные целевой переменной.
    
    Returns:
        model: Обученная модель полносвязной нейронной сети.
        
    Note:
        Гиперпараметры подобраны специфично для проекта."""
        
    model = MLPRegressor(max_iter=1000, learning_rate="invscaling")
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test, X_train, y_train):
    """Оценивает производительность обученной модели.
    Выводит R2-score, MAE-train и RAE-test для переданных модели и выборок."""
    
    print(f"R2-score: {model.score(X_test, y_test)}")
    print(f"MAE-train: {mean_absolute_error(y_train, model.predict(X_train))}\nMAE-test: {mean_absolute_error(y_test, model.predict(X_test))}")
