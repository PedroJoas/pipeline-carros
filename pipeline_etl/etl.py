import requests
from bs4 import BeautifulSoup
import pandas as pd

import re
import logging
from tqdm import tqdm
from pathlib import Path
DADOS_DIR = Path(__file__).resolve().parents[1] / 'dados'

logging.basicConfig(
    filename='pipeline_etl/log_etl.txt',   
    level=logging.INFO,
    format='%(message)s' 
)
class Extrator:

    def __init__(self):
        self.url_base = 'https://www.olx.com.br/autos-e-pecas/carros-vans-e-utilitarios?o={}'
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Referer': 'https://www.google.com',
            'Accept-Language': 'pt-BR,pt;q=0.9,en-US;q=0.8,en;q=0.7',
        }
    
    def __extrai_links_carros(self, qtd_paginas):
        self.links_carros = []
        logging.info('-------------- Começando a extrair os links --------------')
        for num_pagina in range(1, qtd_paginas+1):
            logging.info(f'------ Extraindo links da pagina {num_pagina}------')
            url_formatada = self.url_base.format(num_pagina)
            response = requests.get(url_formatada, headers=self.headers)

            soup  = BeautifulSoup(response.text, 'html.parser')
            for tag in soup.find_all('a', class_='olx-adcard__link'):
                self.links_carros.append(tag['href'])
        logging.info('============ Extração de links finalizada ============')

    def __extrai_info_carros(self):
        class_info_tec = 'ad__sc-2h9gkk-1 bdpQSX olx-d-flex olx-ai-flex-start olx-fd-column olx-flex'
        class_preco = 'ad__sc-q5xder-1 hoJpM'
        infos = []
        logging.info('-------------- Começando a extrair as informações --------------')
        for link in tqdm(self.links_carros):
            dict_temp = {}
            response = requests.get(link, headers=self.headers)
            soup = BeautifulSoup(response.text, 'html.parser')
            try:
                preco = soup.find('div', class_=class_preco).getText(';').split(';')[0][3:]

                dict_temp['precos'] = preco
                informacoes_tecnicas = soup.find_all('div', class_=class_info_tec)
                for info in informacoes_tecnicas:
                    chave, valor = info.getText(separator=';').split(';')
                    dict_temp[chave] = (valor)
            except AttributeError as e:
                logging.info(f'Página não encontrado: {link}')
            infos.append(dict_temp)

        logging.info('============ Extração de links finalizada ============')

        return infos

    def extrair(self, qtd_paginas):
        self.__extrai_links_carros(qtd_paginas)
        dados_brutos = self.__extrai_info_carros()
        
        dados = pd.DataFrame(dados_brutos)

        path = f'{DADOS_DIR}/carros_bruto.csv'
        dados.to_csv(path, index=False)
        logging.info(f'Arquivo salvo em {path}')
    

class Transformador:
    def __init__(self, caminho_df):
        self.df = pd.read_csv(caminho_df, dtype={'precos':str})

    def __converte_tipos(self):
        self.df['precos'] = pd.to_numeric(self.df.precos.str.replace('.',''))
        self.df['Potência do motor'] = pd.to_numeric(self.df['Potência do motor'].apply(
                            lambda x: re.search(r'\d+(\.\d+)?', str(x)).group() if re.search(r'\d+(\.\d+)?', str(x)) else None))
        self.df['Portas'] = pd.to_numeric(self.df['Portas'].str.replace(' Portas', ''))
        logging.info('Colunas convertidas')

    def __trata_nulos(self):
        modas = {
            col: self.df[col].mode()[0]
            for col in self.df.select_dtypes(include='object').columns
            if not self.df[col].mode().empty
        }

        medianas = {
            col: self.df[col].median()
            for col in self.df.select_dtypes(include='number').columns
        }

        preenchimentos = {**modas, **medianas}

        self.df.fillna(preenchimentos, inplace=True)
        logging.info('Valores NaN tratados')

    def transformar(self):
        logging.info('===='*20)
        logging.info('----------- Começando a transformação dos dados -----------')
        self.__converte_tipos()
        self.__trata_nulos()
        path = f'{DADOS_DIR}/carros_transformado.csv'
        self.df.to_csv(path, index=False)    
        logging.info(f'Arquivo salvo em {path}')

if __name__ == '__main__':
    extrator = Extrator()
    extrator.extrair(qtd_paginas=10)

    transformador = Transformador(f'{DADOS_DIR}/carros_bruto.csv')
    transformador.transformar()


