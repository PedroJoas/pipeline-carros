import pandas as pd
import numpy as np
import pickle
import logging


from sklearn.model_selection import RandomizedSearchCV, train_test_split, KFold
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn import metrics

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression

from pathlib import Path

DADOS_DIR = Path(__file__).resolve().parents[1] / 'dados'
DATASET_PATH = f'{DADOS_DIR}/carros_transformado.csv'

def load_dataset(dataset_path) -> pd.DataFrame:
    """
    Carrega um arquivo CSV com métricas extraídas e rótulos associados.

    Parameters:
        dataset_path (str): Caminho para o arquivo CSV do dataset.

    Returns:
        sdp_dataset (pandas.DataFrame): DataFrame com os dados carregados.
    """
    sdp_dataset = pd.read_csv(dataset_path)
    
    return sdp_dataset

def save_model(model, model_name, cv_criteria):
    """
    Salva o modelo fornecido em um arquivo pickle.

    Parameters:
        model (obj): Instância do modelo a ser salva.
        model_name (str): Nome identificador do modelo.
        data_balance (str): Método de balanceamento de dados utilizado.
        cv_criteria (str): Critério de validação cruzada utilizado.
    """      
    with open(f"models/model-{model_name}-{cv_criteria.upper()}.pkl", "wb") as model_file:
        pickle.dump(model, model_file)

def remover_categorias_raras(df, col, limite):
    freq = df[col].value_counts()
    categorias_comuns = freq[freq >= limite].index
    return df[df[col].isin(categorias_comuns)]


def extract_metrics_scores(y_test, y_pred) -> dict:
    """
    Extrai métricas de desempenho de um modelo de regressão.

    Parameters:
        y_test (array-like): Valores reais.
        y_pred (array-like): Valores previstos pelo modelo.

    Returns:
        scores (dict): Dicionário com métricas como RMSE, MAE, R², etc.
    """
    scores = {
        "explained_variance_score": metrics.explained_variance_score(y_test, y_pred),
        "r2_score": metrics.r2_score(y_test, y_pred),
        "mean_absolute_error": metrics.mean_absolute_error(y_test, y_pred),
        "mean_squared_error": metrics.mean_squared_error(y_test, y_pred),
        "root_mean_squared_error": np.sqrt(metrics.mean_squared_error(y_test, y_pred)),
        "median_absolute_error": metrics.median_absolute_error(y_test, y_pred),
        "max_error": metrics.max_error(y_test, y_pred),
        "mean_absolute_percentage_error": metrics.mean_absolute_percentage_error(y_test, y_pred)
    }
    return scores

def treina_kfolds(models, grid_params, x_treino, y_treino):
    categorical_cols = x_treino.select_dtypes(include=['object', 'category']).columns.tolist()
    numerical_cols = x_treino.select_dtypes(include=['int64', 'float64']).columns.tolist()

    kf = KFold(n_splits=10, shuffle=True)
    kf.get_n_splits(x_treino, y_treino)

    models_info_per_fold = {}

    for i, (treino_index, teste_index) in enumerate(kf.split(x_treino, y_treino)):
        x_treino_fold, x_teste_fold = x_treino.iloc[treino_index], x_treino.iloc[teste_index]
        y_treino_fold, y_teste_fold = y_treino.iloc[treino_index], y_treino.iloc[teste_index]

        modelos_info = {}
        for nome_modelo, modelo in models.items():
            passos_preprocessor = [("onehot", OneHotEncoder(handle_unknown='ignore'), categorical_cols)]

            if isinstance(modelo, LinearRegression):
                passos_preprocessor.append(("scaler", StandardScaler(), numerical_cols))
            
            preprocessor = ColumnTransformer(passos_preprocessor)

            pipeline = Pipeline([('preprocessing', preprocessor),
                                 ('modelo', modelo)])

            grid = RandomizedSearchCV(pipeline, grid_params[nome_modelo], n_iter=30, cv=5, scoring='r2', error_score='raise')

            grid.fit(x_treino_fold, y_treino_fold)
            y_pred = grid.predict(x_teste_fold)
            metrics_scores = extract_metrics_scores(y_teste_fold, y_pred)

            modelos_info[nome_modelo] = {
                "score": metrics_scores,
                "best_estimator": grid.best_estimator_
            }
        models_info_per_fold[i] = modelos_info

    return models_info_per_fold


def do_benchmark(x_treino, y_treino) -> dict:    
    """
    Orquestra o benchmark de modelos selecionados a partir de um arquivo de dados.

    Parameters:
        grid_search (bool): Se True, utiliza Randomizer para otimização de hiperparâmetros.
        data_balance (str): Método de balanceamento de dados ('SMOTE' ou outro).
        dataset_path (str): Caminho para o arquivo CSV do dataset.
        cv_criteria (str): Critério de scoring para validação cruzada (ex: 'roc_auc').
        selected_models (list of str): Lista de chaves dos modelos a serem treinados.

    Returns:
        fold_results (dict): Resultado do benchmark para cada fold.
    """    

    
    treino_models = {
                "GBR": GradientBoostingRegressor(),
                "RFR": RandomForestRegressor(),
                "LR": LinearRegression()
                }

    grid_params_list = {
                    "GBR": {
                        "modelo__n_estimators": [50, 100, 150, 200],
                        "modelo__learning_rate": [0.01, 0.05, 0.1, 0.2],
                        "modelo__max_depth": [3, 4, 5, None]
                        },
                    "RFR": {
                        "modelo__n_estimators": [50, 100, 150],
                        "modelo__max_depth": [4, 5, 6, None],
                        "modelo__max_features": [None, "sqrt", "log2"]
                    },
                    "LR": {
                        "modelo__fit_intercept": [True, False]
                    }
                }


    
    fold_results = treina_kfolds(x_treino=x_treino, 
                                  y_treino=y_treino, 
                                  models=treino_models, grid_params=grid_params_list)

    return fold_results

def select_best_model(fold_results):
    results = {}
    for fold in fold_results:
        for model_name, model_data in fold_results[fold].items():
            r2 = model_data["score"]["r2_score"]
            if model_name not in results:
                results[model_name] = {"r2_score": [], "estimators": []}
            results[model_name]["r2_score"].append(r2)
            results[model_name]["estimators"].append((fold, model_data['best_estimator']))

    best_model_name = None
    best_model_score = float("-inf")
    best_fold = None
    for model_name, data in results.items():
        avg_r2 = sum(data["r2_score"]) / len(data["r2_score"])
        if avg_r2 > best_model_score:
            best_model_score = avg_r2
            best_model_name = model_name
            
            best_fold_idx = data["r2_score"].index(max(data["r2_score"]))
            best_fold = data["estimators"][best_fold_idx][0]

    best_estimator = fold_results[best_fold][best_model_name]['best_estimator']

    return best_model_name, best_estimator


def build_champion_model(x_treino, y_treino, x_teste, y_teste, model_info, best_estimator) -> dict:
    """
    Treina o pipeline final com todos os dados de treino (sem busca de hiperparâmetros)
    e avalia no conjunto de teste.

    Parameters:
        model_info (dict): Info do modelo (nome, etc).
        best_estimator (Pipeline): Pipeline já configurado (com hiperparâmetros).

    Returns:
        metrics_scores (dict): Métricas de desempenho final.
    """    
    
    best_estimator.fit(x_treino, y_treino)

    y_pred = best_estimator.predict(x_teste)

    metrics_scores = extract_metrics_scores(y_teste, y_pred)

    save_model(best_estimator, model_info.get("name"), cv_criteria="r2")

    return metrics_scores


def make_model(x_treino, y_treino, x_teste, y_teste, selected_model=None, best_estimator=None) -> dict:    
    """
    Constrói e avalia o modelo selecionado como champion.

    Parameters:
        grid_search (bool): Se True, usa Randomizer para otimizar hiperparâmetros.
        data_balance (str): Método de balanceamento de dados ('SMOTE' ou outro).
        dataset_path (str): Caminho para o arquivo CSV do dataset.
        cv_criteria (str): Critério de scoring para validação cruzada (ex: 'roc_auc').
        selected_model (str): Chave do modelo a treinar (e.g., 'LRC', 'RFC', 'SVC').

    Returns:
        metrics_scores (dict): Métricas de desempenho do modelo final.
    """    
    
    treino_models = {
                "GBR": GradientBoostingRegressor(),
                "RFR": RandomForestRegressor(),
                "LR": LinearRegression()
                }

    grid_params_list = {
                    "GBR": {
                        "modelo__n_estimators": [50, 100, 150, 200],
                        "modelo__learning_rate": [0.01, 0.05, 0.1, 0.2],
                        "modelo__max_depth": [3, 4, 5, None]
                        },
                    "RFR": {
                        "modelo__n_estimators": [50, 100, 150],
                        "modelo__max_depth": [4, 5, 6, None],
                        "modelo__max_features": [None, "sqrt", "log2"]
                    },
                    "LR": {
                        "modelo__fit_intercept": [True, False]
                    }
                }
    
    model_info = {
        "name": selected_model,
        "instance": treino_models.get(selected_model),
        "grid_params_list":grid_params_list.get(selected_model)
    }    

    metrics_scores = build_champion_model(x_treino, y_treino, x_teste, y_teste, 
                                          model_info, best_estimator)
    return metrics_scores

def start(dataset_path):
    logger = logging.getLogger(__name__)
    logging.basicConfig(filename='pipeline_ml/pipeline_ml.log', encoding='utf-8', level=logging.DEBUG)
    logger.debug("[Step-1] Realizando Benchmark")
    dataset = pd.read_csv(dataset_path)
    logger.debug("[Step-1.1] Tratando Dataset")

    dataset.drop(['Categoria', 'Modelo', 'Possui Kit GNV', 'Final de placa', 'Cor'], axis=1, inplace=True)

    colunas = ['Marca', 'Tipo de veículo', 'Potência do motor', 
        'Combustível', 'Câmbio', 'Direção', 'Portas', 'Tipo de direção']

    for coluna in colunas:
        dataset = remover_categorias_raras(dataset, coluna, limite=10)

    X = dataset.drop('precos', axis=1)
    y = dataset.precos

    x_treino, x_teste, y_treino, y_teste = train_test_split(X, y, test_size=0.2, random_state=42)
      
    fold_results = do_benchmark(x_treino,y_treino)
    
    for fold, models_info in fold_results.items():
        for model_name, info in models_info.items():
            sc = info['score']
            logger.debug(
                f"Fold {fold} - Model {model_name}: "
                f"r2={sc['r2_score']:.4f}, EVS={sc['explained_variance_score']:.4f}, "
                f"MAE={sc['mean_absolute_error']:.2f}, MSE={sc['mean_squared_error']:.2f}, "
                f"RMSE={sc['root_mean_squared_error']:.2f}, "
                f"MedAE={sc['median_absolute_error']:.2f}, "
                f"MaxErr={sc['max_error']:.2f}, "
                f"MAPE={sc['mean_absolute_percentage_error']:.2%}"
            )

    
    logger.debug("\n[Step-2] Selecionando Melhor Modelo")
    best_model_name, best_estimator = select_best_model(fold_results)
    logger.debug(f"\nBest Model {best_model_name}")
    
    logger.debug("\n[Step-3] Criando o Modelo Final")
    metric_scores = make_model(x_treino, y_treino, x_teste, y_teste, 
                               selected_model=best_model_name, best_estimator=best_estimator)
    sc = metric_scores
    logger.debug(
        f"Champion {best_model_name}: r2={sc['r2_score']}, "
        f"MAE={sc['mean_absolute_error']}, MSE={sc['mean_squared_error']}, "
        f"RMSE={sc['root_mean_squared_error']}, MedAE={sc['median_absolute_error']}, "
        f"MaxErr={sc['max_error']}, MedMAPEAE={sc['mean_absolute_percentage_error']}, "
    )

if __name__ == '__main__':  
    start(DATASET_PATH)

