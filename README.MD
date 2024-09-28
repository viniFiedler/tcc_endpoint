
# CãoAPI - Sistema de Identificação de Cães

## Descrição

CãoAPI é uma API desenvolvida em Flask para registrar e identificar cães em imagens. Utiliza um modelo YOLO para segmentar a imagem e um modelo ViT (Vision Transformer) para extrair vetores de características e comparar com registros já armazenados.

### Principais Funcionalidades

1. **Registrar Imagem**: A API recebe uma imagem de um cachorro, realiza a segmentação da imagem com o YOLO e extrai um vetor de características com o ViT.
2. **Identificar Imagem**: Com base em uma nova imagem enviada, a API segmenta o cachorro e compara o vetor de características com os registros existentes, retornando o registro mais próximo.
3. **Documentação da API**: A página inicial da API explica o funcionamento das rotas.

## Tecnologias Utilizadas

- **Flask**: Framework para desenvolvimento da API.
- **YOLOv8**: Modelo para detecção e segmentação de objetos.
- **Vision Transformer (ViT)**: Modelo da Hugging Face para extração de descritores de imagem.
- **PIL**: Biblioteca Python para processamento de imagens.
- **Gunicorn**: Servidor WSGI para a API Flask.
- **Nginx**: Proxy reverso para a aplicação.

## Pré-requisitos

- Python 3.8 ou superior
- Pip (gerenciador de pacotes Python)
- Ambiente virtual Python (recomendado)
- Modelos YOLO e ViT já treinados

## Como Instalar

1. **Clone o repositório:**
   ```bash
   git clone https://github.com/seu-usuario/caoapi.git
   cd caoapi
   ```

2. **Crie e ative um ambiente virtual:**
   ```bash
   python3 -m venv myenv
   source myenv/bin/activate  # Para Linux/Mac
   # ou
   myenv\Scripts ctivate  # Para Windows
   ```

3. **Instale as dependências:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Baixe os modelos YOLO e ViT:**
   - YOLO: Coloque o arquivo `best.pt` na raiz do projeto.
   - ViT: Certifique-se de que o modelo `vit-base-patch16-224-finetuned-eurosat` está disponível no Hugging Face.

## Como Usar

### 1. Executar a API localmente:

```bash
gunicorn -w 3 --bind 0.0.0.0:8000 wsgi:app
```

### 2. Roteamento:

#### Registrar uma nova imagem
- **Endpoint**: `/registrar`
- **Método**: `POST`
- **Parâmetros**: 
  - `file`: A imagem a ser registrada.
  - `nome`: O nome do cachorro.
- **Exemplo de resposta**:
  ```json
  {
    "message": "Registro salvo!",
    "id": "UUID-gerado",
    "nome": "Nome do cachorro"
  }
  ```

#### Identificar uma imagem
- **Endpoint**: `/identificar`
- **Método**: `POST`
- **Parâmetros**:
  - `file`: A imagem do cachorro a ser identificada.
- **Exemplo de resposta**:
  ```json
  {
    "message": "Registro mais próximo encontrado",
    "registro": {
      "id": "UUID-do-registro",
      "nome": "Nome do cachorro"
    }
  }
  ```

## Deploy no Servidor

1. **Configuração do Gunicorn e Supervisor**:
   Certifique-se de que o Supervisor está configurado para rodar o Gunicorn.

2. **Configuração do Nginx**:
   O Nginx atua como proxy reverso, apontando para o Gunicorn que está rodando a API.

## Estrutura do Projeto

```
/caoapi
│
├── /static             # Arquivos estáticos (imagens, etc)
├── /templates          # Templates HTML
├── /models             # Modelos YOLO e ViT
├── wsgi.py             # Ponto de entrada do WSGI
├── app.py              # Arquivo principal da API Flask
├── TrataImagems.py     # Lógica de segmentação YOLO
├── PaginaInicial.py    # Página inicial da API
└── requirements.txt    # Dependências do projeto
```

## Contato

- Autor: Vinicius Garcia
- Email: fiedler.vinicius1@gmail.com
