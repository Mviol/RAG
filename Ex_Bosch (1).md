#### Resumo

O código começa importando as bibliotecas necessárias (fpdf e random).
Define uma função gerar_dados() que cria 10.000 linhas de dados aleatórios sobre atendimentos, vendas, reclamações e elogios para diferentes meses.
Cria um objeto PDF usando FPDF.
Adiciona 5 páginas de texto introdutório sobre a Bosch.
Gera uma grande tabela com os 10.000 registros de dados fictícios.
Salva o PDF como 'relatorio_bosch.pdf'.
Extração de texto e imagens do PDF:

Usa a biblioteca PyMuPDF (fitz) para abrir e processar o PDF.
Define uma função extract_text_and_images_from_pdf() que extrai todo o texto e imagens do PDF.
Imprime uma amostra do texto extraído e o número de imagens encontradas.
Divisão do texto em chunks:

Importa o RecursiveCharacterTextSplitter da biblioteca langchain.
Define um texto de exemplo.
Configura o splitter para criar chunks de 500 caracteres com 50 caracteres de sobreposição.
Divide o texto em chunks e os imprime.
Configuração de modelos de linguagem e embeddings:

Importa um tokenizer e modelo GPT-J da biblioteca transformers (comentado no código).
Configura um sistema de embeddings usando a biblioteca txtai.
Cálculo de similaridade do cosseno:

Importa numpy para cálculos vetoriais.
Define uma função cosine_similarity() que calcula a similaridade do cosseno entre dois vetores de embedding.
Demonstra o uso da função com dois vetores de exemplo.

# TESTE


```python
from fpdf import FPDF
import random

# Função para gerar dados fictícios
def gerar_dados():
    meses = ["Janeiro", "Fevereiro", "Março", "Abril", "Maio", "Junho",
             "Julho", "Agosto", "Setembro", "Outubro", "Novembro", "Dezembro"]
    dados = []
    for i in range(10000):
        mes = random.choice(meses)
        atendimentos = random.randint(100, 300)
        vendas = random.randint(2000, 5000)
        reclamacoes = random.randint(5, 50)
        elogios = random.randint(1, 30)
        dados.append([mes, atendimentos, vendas, reclamacoes, elogios])
    return dados

# Criando o documento PDF
pdf = FPDF()

# Texto introdutório (5 páginas)
pdf.set_font('Arial', '', 12)
for i in range(5):
    pdf.add_page()
    pdf.multi_cell(0, 10, f"""
    Página {i+1}
    A Bosch é líder mundial no fornecimento de tecnologia e serviços. A empresa emprega aproximadamente 400.000 
    colaboradores em todo o mundo. Neste relatório, apresentamos um resumo das atividades de atendimento ao cliente, 
    vendas, reclamações e elogios, bem como gráficos e análises estatísticas dos dados. Este documento contém informações
    detalhadas sobre o desempenho da empresa durante os últimos 12 meses. Inclui métricas de desempenho e insights valiosos
    para entender melhor as tendências do mercado.
    
    Informação adicional: A Bosch é conhecida por sua inovação e qualidade. Nossa equipe está comprometida em fornecer 
    as melhores soluções tecnológicas para nossos clientes. Estamos sempre à frente das tendências do mercado, garantindo 
    a satisfação e confiança de nossos consumidores.
    """)

# Gerando tabela grande com 10.000 linhas de dados fictícios
dados = gerar_dados()

# Adicionando a tabela ao PDF
pdf.add_page()
pdf.set_font('Arial', 'B', 12)
pdf.cell(30, 10, 'Mês', 1)
pdf.cell(40, 10, 'Atendimentos', 1)
pdf.cell(40, 10, 'Vendas', 1)
pdf.cell(40, 10, 'Reclamações', 1)
pdf.cell(40, 10, 'Elogios', 1)
pdf.ln(10)
pdf.set_font('Arial', '', 12)

# Loop para adicionar todas as linhas da tabela
for linha in dados:
    for item in linha:
        pdf.cell(30, 10, str(item), 1)
    pdf.ln(10)

# Salvando o PDF
pdf.output('relatorio_bosch.pdf')

```




    ''




```python
import fitz  # PyMuPDF

def extract_text_and_images_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    images = []
    for page in doc:
        text += page.get_text()
        for img_index, img in enumerate(page.get_images(full=True)):
            xref = img[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            image_ext = base_image["ext"]
            images.append((image_bytes, image_ext, img_index))
    return text, images

# Caminho do arquivo PDF
pdf_path = 'relatorio_bosch.pdf'
pdf_text, pdf_images = extract_text_and_images_from_pdf(pdf_path)

print("Texto extraído:")
print(pdf_text[:1000])  # Mostrar os primeiros 1000 caracteres do texto extraído
print(f"Número de imagens extraídas: {len(pdf_images)}")

```

    Texto extraído:
        Página 1
        A Bosch é líder mundial no fornecimento de tecnologia e serviços. A empresa emprega
    aproximadamente 400.000 
        colaboradores em todo o mundo. Neste relatório, apresentamos um resumo das atividades de
    atendimento ao cliente, 
        vendas, reclamações e elogios, bem como gráficos e análises estatísticas dos dados. Este
    documento contém informações
        detalhadas sobre o desempenho da empresa durante os últimos 12 meses. Inclui métricas de
    desempenho e insights valiosos
        para entender melhor as tendências do mercado.
        
        Informação adicional: A Bosch é conhecida por sua inovação e qualidade. Nossa equipe está
    comprometida em fornecer 
        as melhores soluções tecnológicas para nossos clientes. Estamos sempre à frente das
    tendências do mercado, garantindo 
        a satisfação e confiança de nossos consumidores.
        
        Página 2
        A Bosch é líder mundial no fornecimento de tecnologia e serviços. A empresa emprega
    aproximadamente 400.000 
        colaboradores em todo o
    Número de imagens extraídas: 0
    


```python
def create_chunks(text, chunk_size=100):
    words = text.split()
    return
```

#### Funcionalidade para Extrair os Dados do Documento:


```python
import fitz  # PyMuPDF

def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page_num in range(doc.page_count):
        page = doc.load_page(page_num)
        text += page.get_text()
    return text

```

####  LLM Open Source e/ou na Nuvem:


```python
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B")
model = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-j-6B")

```


    pytorch_model.bin:  29%|##8       | 7.01G/24.2G [00:00<?, ?B/s]


    Error while downloading from https://cdn-lfs.hf.co/EleutherAI/gpt-j-6B/0e183edc2025ecfdba4429ba43c960224103b3c3dc26616503cdc2158a3d6c93?response-content-disposition=inline%3B+filename*%3DUTF-8%27%27pytorch_model.bin%3B+filename%3D%22pytorch_model.bin%22%3B&response-content-type=application%2Foctet-stream&Expires=1729265152&Policy=eyJTdGF0ZW1lbnQiOlt7IkNvbmRpdGlvbiI6eyJEYXRlTGVzc1RoYW4iOnsiQVdTOkVwb2NoVGltZSI6MTcyOTI2NTE1Mn19LCJSZXNvdXJjZSI6Imh0dHBzOi8vY2RuLWxmcy5oZi5jby9FbGV1dGhlckFJL2dwdC1qLTZCLzBlMTgzZWRjMjAyNWVjZmRiYTQ0MjliYTQzYzk2MDIyNDEwM2IzYzNkYzI2NjE2NTAzY2RjMjE1OGEzZDZjOTM%7EcmVzcG9uc2UtY29udGVudC1kaXNwb3NpdGlvbj0qJnJlc3BvbnNlLWNvbnRlbnQtdHlwZT0qIn1dfQ__&Signature=BmJMUCH9nwztT3hjaIDBsd9qlD2nXU9E%7EVw4C84MQV9Evwyq3WGWKz5JSaIqbCDRfGq9GZj%7EVWVknLHtPLZOBa13iQaxYtHpXPKnV7IPYZvWZ7jpAZ3L5Uv148TnFcmnxsb%7EZoRFKwF%7E8tb7ND4U7GN0i%7EWgZYjrenyXJSbS3Oard8%7Ew9EqytXNoAzSRFyZYtvqiuSjrZbzEYkmFOFcGdjKi5nSESl32NEzQViOsDL1fwfhTAgOUmj6wmvHRXKg2cjnHPSTORuCZ1baQIGvSB2-kkVczIPMiHNigPTGCS5rzPJJBOCRqEqEUXS-9yneDffYntGrqhPFIz6GtaMuT8Q__&Key-Pair-Id=K3RPWS32NSSJCE: HTTPSConnectionPool(host='cdn-lfs.hf.co', port=443): Read timed out.
    Trying to resume download...
    


    pytorch_model.bin:  93%|#########2| 22.4G/24.2G [00:00<?, ?B/s]


    C:\Users\mttvi\anaconda3\lib\site-packages\torch\_utils.py:831: UserWarning: TypedStorage is deprecated. It will be removed in the future and UntypedStorage will be the only storage class. This should only matter to you if you are using storages directly.  To access UntypedStorage directly, use tensor.untyped_storage() instead of tensor.storage()
      return self.fget.__get__(instance, owner)()
    

####  Base de Dados para os Embeddings:


```python
pip install txtai
```

    Collecting txtai
      Downloading txtai-7.5.0-py3-none-any.whl (244 kB)
         ------------------------------------ 244.6/244.6 kB 937.5 kB/s eta 0:00:00
    Requirement already satisfied: huggingface-hub>=0.19.0 in c:\users\mttvi\anaconda3\lib\site-packages (from txtai) (0.25.2)
    Collecting regex>=2022.8.17
      Downloading regex-2024.9.11-cp39-cp39-win_amd64.whl (274 kB)
         -------------------------------------- 274.1/274.1 kB 1.7 MB/s eta 0:00:00
    Collecting msgpack>=1.0.7
      Downloading msgpack-1.1.0-cp39-cp39-win_amd64.whl (74 kB)
         ---------------------------------------- 74.8/74.8 kB 1.0 MB/s eta 0:00:00
    Requirement already satisfied: faiss-cpu>=1.7.1.post2 in c:\users\mttvi\anaconda3\lib\site-packages (from txtai) (1.9.0)
    Requirement already satisfied: transformers>=4.28.0 in c:\users\mttvi\anaconda3\lib\site-packages (from txtai) (4.45.2)
    Requirement already satisfied: torch>=1.12.1 in c:\users\mttvi\anaconda3\lib\site-packages (from txtai) (2.1.2)
    Requirement already satisfied: numpy>=1.18.4 in c:\users\mttvi\anaconda3\lib\site-packages (from txtai) (2.0.2)
    Requirement already satisfied: pyyaml>=5.3 in c:\users\mttvi\anaconda3\lib\site-packages (from txtai) (6.0)
    Requirement already satisfied: packaging in c:\users\mttvi\anaconda3\lib\site-packages (from faiss-cpu>=1.7.1.post2->txtai) (21.3)
    Requirement already satisfied: typing-extensions>=3.7.4.3 in c:\users\mttvi\anaconda3\lib\site-packages (from huggingface-hub>=0.19.0->txtai) (4.12.2)
    Requirement already satisfied: tqdm>=4.42.1 in c:\users\mttvi\anaconda3\lib\site-packages (from huggingface-hub>=0.19.0->txtai) (4.64.1)
    Requirement already satisfied: fsspec>=2023.5.0 in c:\users\mttvi\anaconda3\lib\site-packages (from huggingface-hub>=0.19.0->txtai) (2024.9.0)
    Requirement already satisfied: requests in c:\users\mttvi\anaconda3\lib\site-packages (from huggingface-hub>=0.19.0->txtai) (2.28.1)
    Requirement already satisfied: filelock in c:\users\mttvi\anaconda3\lib\site-packages (from huggingface-hub>=0.19.0->txtai) (3.6.0)
    Requirement already satisfied: sympy in c:\users\mttvi\anaconda3\lib\site-packages (from torch>=1.12.1->txtai) (1.10.1)
    Requirement already satisfied: networkx in c:\users\mttvi\anaconda3\lib\site-packages (from torch>=1.12.1->txtai) (2.8.4)
    Requirement already satisfied: jinja2 in c:\users\mttvi\anaconda3\lib\site-packages (from torch>=1.12.1->txtai) (2.11.3)
    Requirement already satisfied: tokenizers<0.21,>=0.20 in c:\users\mttvi\anaconda3\lib\site-packages (from transformers>=4.28.0->txtai) (0.20.1)
    Requirement already satisfied: safetensors>=0.4.1 in c:\users\mttvi\anaconda3\lib\site-packages (from transformers>=4.28.0->txtai) (0.4.5)
    Requirement already satisfied: pyparsing!=3.0.5,>=2.0.2 in c:\users\mttvi\anaconda3\lib\site-packages (from packaging->faiss-cpu>=1.7.1.post2->txtai) (3.0.9)
    Requirement already satisfied: colorama in c:\users\mttvi\anaconda3\lib\site-packages (from tqdm>=4.42.1->huggingface-hub>=0.19.0->txtai) (0.4.5)
    Requirement already satisfied: MarkupSafe>=0.23 in c:\users\mttvi\anaconda3\lib\site-packages (from jinja2->torch>=1.12.1->txtai) (2.0.1)
    Requirement already satisfied: certifi>=2017.4.17 in c:\users\mttvi\anaconda3\lib\site-packages (from requests->huggingface-hub>=0.19.0->txtai) (2022.9.14)
    Requirement already satisfied: charset-normalizer<3,>=2 in c:\users\mttvi\anaconda3\lib\site-packages (from requests->huggingface-hub>=0.19.0->txtai) (2.0.4)
    Requirement already satisfied: idna<4,>=2.5 in c:\users\mttvi\anaconda3\lib\site-packages (from requests->huggingface-hub>=0.19.0->txtai) (3.3)
    Requirement already satisfied: urllib3<1.27,>=1.21.1 in c:\users\mttvi\anaconda3\lib\site-packages (from requests->huggingface-hub>=0.19.0->txtai) (1.26.11)
    Requirement already satisfied: mpmath>=0.19 in c:\users\mttvi\anaconda3\lib\site-packages (from sympy->torch>=1.12.1->txtai) (1.2.1)
    Installing collected packages: regex, msgpack, txtai
      Attempting uninstall: regex
        Found existing installation: regex 2022.7.9
        Uninstalling regex-2022.7.9:
          Successfully uninstalled regex-2022.7.9
      Attempting uninstall: msgpack
        Found existing installation: msgpack 1.0.3
        Uninstalling msgpack-1.0.3:
          Successfully uninstalled msgpack-1.0.3
    Successfully installed msgpack-1.1.0 regex-2024.9.11 txtai-7.5.0
    Note: you may need to restart the kernel to use updated packages.
    

    WARNING: Ignoring invalid distribution -otobuf (c:\users\mttvi\anaconda3\lib\site-packages)
    WARNING: Ignoring invalid distribution -rotobuf (c:\users\mttvi\anaconda3\lib\site-packages)
    WARNING: Ignoring invalid distribution - (c:\users\mttvi\anaconda3\lib\site-packages)
    WARNING: Ignoring invalid distribution -otobuf (c:\users\mttvi\anaconda3\lib\site-packages)
    WARNING: Ignoring invalid distribution -rotobuf (c:\users\mttvi\anaconda3\lib\site-packages)
    WARNING: Ignoring invalid distribution - (c:\users\mttvi\anaconda3\lib\site-packages)
    WARNING: Ignoring invalid distribution -otobuf (c:\users\mttvi\anaconda3\lib\site-packages)
    WARNING: Ignoring invalid distribution -rotobuf (c:\users\mttvi\anaconda3\lib\site-packages)
    WARNING: Ignoring invalid distribution - (c:\users\mttvi\anaconda3\lib\site-packages)
        WARNING: Ignoring invalid distribution -otobuf (c:\users\mttvi\anaconda3\lib\site-packages)
        WARNING: Ignoring invalid distribution -rotobuf (c:\users\mttvi\anaconda3\lib\site-packages)
        WARNING: Ignoring invalid distribution - (c:\users\mttvi\anaconda3\lib\site-packages)
        WARNING: Ignoring invalid distribution -otobuf (c:\users\mttvi\anaconda3\lib\site-packages)
        WARNING: Ignoring invalid distribution -rotobuf (c:\users\mttvi\anaconda3\lib\site-packages)
        WARNING: Ignoring invalid distribution - (c:\users\mttvi\anaconda3\lib\site-packages)
    WARNING: Ignoring invalid distribution -otobuf (c:\users\mttvi\anaconda3\lib\site-packages)
    WARNING: Ignoring invalid distribution -rotobuf (c:\users\mttvi\anaconda3\lib\site-packages)
    WARNING: Ignoring invalid distribution - (c:\users\mttvi\anaconda3\lib\site-packages)
    WARNING: Ignoring invalid distribution -otobuf (c:\users\mttvi\anaconda3\lib\site-packages)
    WARNING: Ignoring invalid distribution -rotobuf (c:\users\mttvi\anaconda3\lib\site-packages)
    WARNING: Ignoring invalid distribution - (c:\users\mttvi\anaconda3\lib\site-packages)
    WARNING: Ignoring invalid distribution -otobuf (c:\users\mttvi\anaconda3\lib\site-packages)
    WARNING: Ignoring invalid distribution -rotobuf (c:\users\mttvi\anaconda3\lib\site-packages)
    WARNING: Ignoring invalid distribution - (c:\users\mttvi\anaconda3\lib\site-packages)
    WARNING: Ignoring invalid distribution -otobuf (c:\users\mttvi\anaconda3\lib\site-packages)
    WARNING: Ignoring invalid distribution -rotobuf (c:\users\mttvi\anaconda3\lib\site-packages)
    WARNING: Ignoring invalid distribution - (c:\users\mttvi\anaconda3\lib\site-packages)
    WARNING: Ignoring invalid distribution -otobuf (c:\users\mttvi\anaconda3\lib\site-packages)
    WARNING: Ignoring invalid distribution -rotobuf (c:\users\mttvi\anaconda3\lib\site-packages)
    WARNING: Ignoring invalid distribution - (c:\users\mttvi\anaconda3\lib\site-packages)
    WARNING: Ignoring invalid distribution -otobuf (c:\users\mttvi\anaconda3\lib\site-packages)
    WARNING: Ignoring invalid distribution -rotobuf (c:\users\mttvi\anaconda3\lib\site-packages)
    WARNING: Ignoring invalid distribution - (c:\users\mttvi\anaconda3\lib\site-packages)
    


```python
from txtai.embeddings import Embeddings

embeddings = Embeddings()
embeddings.index([("1", "Text data to index")])
```

    
    A module that was compiled using NumPy 1.x cannot be run in
    NumPy 2.0.2 as it may crash. To support both 1.x and 2.x
    versions of NumPy, modules must be compiled with NumPy 2.0.
    Some module may need to rebuild instead e.g. with 'pybind11>=2.12'.
    
    If you are a user of the module, the easiest solution will be to
    downgrade to 'numpy<2' or try to upgrade the affected module.
    We expect that some modules will need time to support NumPy 2.
    
    Traceback (most recent call last):  File "C:\Users\mttvi\anaconda3\lib\runpy.py", line 197, in _run_module_as_main
        return _run_code(code, main_globals, None,
      File "C:\Users\mttvi\anaconda3\lib\runpy.py", line 87, in _run_code
        exec(code, run_globals)
      File "C:\Users\mttvi\anaconda3\lib\site-packages\ipykernel_launcher.py", line 17, in <module>
        app.launch_new_instance()
      File "C:\Users\mttvi\anaconda3\lib\site-packages\traitlets\config\application.py", line 846, in launch_instance
        app.start()
      File "C:\Users\mttvi\anaconda3\lib\site-packages\ipykernel\kernelapp.py", line 712, in start
        self.io_loop.start()
      File "C:\Users\mttvi\anaconda3\lib\site-packages\tornado\platform\asyncio.py", line 199, in start
        self.asyncio_loop.run_forever()
      File "C:\Users\mttvi\anaconda3\lib\asyncio\base_events.py", line 601, in run_forever
        self._run_once()
      File "C:\Users\mttvi\anaconda3\lib\asyncio\base_events.py", line 1905, in _run_once
        handle._run()
      File "C:\Users\mttvi\anaconda3\lib\asyncio\events.py", line 80, in _run
        self._context.run(self._callback, *self._args)
      File "C:\Users\mttvi\anaconda3\lib\site-packages\ipykernel\kernelbase.py", line 510, in dispatch_queue
        await self.process_one()
      File "C:\Users\mttvi\anaconda3\lib\site-packages\ipykernel\kernelbase.py", line 499, in process_one
        await dispatch(*args)
      File "C:\Users\mttvi\anaconda3\lib\site-packages\ipykernel\kernelbase.py", line 406, in dispatch_shell
        await result
      File "C:\Users\mttvi\anaconda3\lib\site-packages\ipykernel\kernelbase.py", line 730, in execute_request
        reply_content = await reply_content
      File "C:\Users\mttvi\anaconda3\lib\site-packages\ipykernel\ipkernel.py", line 390, in do_execute
        res = shell.run_cell(code, store_history=store_history, silent=silent)
      File "C:\Users\mttvi\anaconda3\lib\site-packages\ipykernel\zmqshell.py", line 528, in run_cell
        return super().run_cell(*args, **kwargs)
      File "C:\Users\mttvi\anaconda3\lib\site-packages\IPython\core\interactiveshell.py", line 2914, in run_cell
        result = self._run_cell(
      File "C:\Users\mttvi\anaconda3\lib\site-packages\IPython\core\interactiveshell.py", line 2960, in _run_cell
        return runner(coro)
      File "C:\Users\mttvi\anaconda3\lib\site-packages\IPython\core\async_helpers.py", line 78, in _pseudo_sync_runner
        coro.send(None)
      File "C:\Users\mttvi\anaconda3\lib\site-packages\IPython\core\interactiveshell.py", line 3185, in run_cell_async
        has_raised = await self.run_ast_nodes(code_ast.body, cell_name,
      File "C:\Users\mttvi\anaconda3\lib\site-packages\IPython\core\interactiveshell.py", line 3377, in run_ast_nodes
        if (await self.run_code(code, result,  async_=asy)):
      File "C:\Users\mttvi\anaconda3\lib\site-packages\IPython\core\interactiveshell.py", line 3457, in run_code
        exec(code_obj, self.user_global_ns, self.user_ns)
      File "C:\Users\mttvi\AppData\Local\Temp\ipykernel_15100\3706768229.py", line 1, in <module>
        from txtai.embeddings import Embeddings
      File "C:\Users\mttvi\anaconda3\lib\site-packages\txtai\__init__.py", line 8, in <module>
        from .app import Application
      File "C:\Users\mttvi\anaconda3\lib\site-packages\txtai\app\__init__.py", line 5, in <module>
        from .base import Application, ReadOnlyError
      File "C:\Users\mttvi\anaconda3\lib\site-packages\txtai\app\base.py", line 12, in <module>
        from ..embeddings import Documents, Embeddings
      File "C:\Users\mttvi\anaconda3\lib\site-packages\txtai\embeddings\__init__.py", line 5, in <module>
        from .base import Embeddings
      File "C:\Users\mttvi\anaconda3\lib\site-packages\txtai\embeddings\base.py", line 12, in <module>
        from ..ann import ANNFactory
      File "C:\Users\mttvi\anaconda3\lib\site-packages\txtai\ann\__init__.py", line 7, in <module>
        from .factory import ANNFactory
      File "C:\Users\mttvi\anaconda3\lib\site-packages\txtai\ann\factory.py", line 13, in <module>
        from .torch import Torch
      File "C:\Users\mttvi\anaconda3\lib\site-packages\txtai\ann\torch.py", line 6, in <module>
        import torch
      File "C:\Users\mttvi\anaconda3\lib\site-packages\torch\__init__.py", line 1382, in <module>
        from .functional import *  # noqa: F403
      File "C:\Users\mttvi\anaconda3\lib\site-packages\torch\functional.py", line 7, in <module>
        import torch.nn.functional as F
      File "C:\Users\mttvi\anaconda3\lib\site-packages\torch\nn\__init__.py", line 1, in <module>
        from .modules import *  # noqa: F403
      File "C:\Users\mttvi\anaconda3\lib\site-packages\torch\nn\modules\__init__.py", line 35, in <module>
        from .transformer import TransformerEncoder, TransformerDecoder, \
      File "C:\Users\mttvi\anaconda3\lib\site-packages\torch\nn\modules\transformer.py", line 20, in <module>
        device: torch.device = torch.device(torch._C._get_default_device()),  # torch.device('cpu'),
    C:\Users\mttvi\anaconda3\lib\site-packages\torch\nn\modules\transformer.py:20: UserWarning: Failed to initialize NumPy: _ARRAY_API not found (Triggered internally at C:\actions-runner\_work\pytorch\pytorch\builder\windows\pytorch\torch\csrc\utils\tensor_numpy.cpp:84.)
      device: torch.device = torch.device(torch._C._get_default_device()),  # torch.device('cpu'),
    
    A module that was compiled using NumPy 1.x cannot be run in
    NumPy 2.0.2 as it may crash. To support both 1.x and 2.x
    versions of NumPy, modules must be compiled with NumPy 2.0.
    Some module may need to rebuild instead e.g. with 'pybind11>=2.12'.
    
    If you are a user of the module, the easiest solution will be to
    downgrade to 'numpy<2' or try to upgrade the affected module.
    We expect that some modules will need time to support NumPy 2.
    
    Traceback (most recent call last):  File "C:\Users\mttvi\anaconda3\lib\runpy.py", line 197, in _run_module_as_main
        return _run_code(code, main_globals, None,
      File "C:\Users\mttvi\anaconda3\lib\runpy.py", line 87, in _run_code
        exec(code, run_globals)
      File "C:\Users\mttvi\anaconda3\lib\site-packages\ipykernel_launcher.py", line 17, in <module>
        app.launch_new_instance()
      File "C:\Users\mttvi\anaconda3\lib\site-packages\traitlets\config\application.py", line 846, in launch_instance
        app.start()
      File "C:\Users\mttvi\anaconda3\lib\site-packages\ipykernel\kernelapp.py", line 712, in start
        self.io_loop.start()
      File "C:\Users\mttvi\anaconda3\lib\site-packages\tornado\platform\asyncio.py", line 199, in start
        self.asyncio_loop.run_forever()
      File "C:\Users\mttvi\anaconda3\lib\asyncio\base_events.py", line 601, in run_forever
        self._run_once()
      File "C:\Users\mttvi\anaconda3\lib\asyncio\base_events.py", line 1905, in _run_once
        handle._run()
      File "C:\Users\mttvi\anaconda3\lib\asyncio\events.py", line 80, in _run
        self._context.run(self._callback, *self._args)
      File "C:\Users\mttvi\anaconda3\lib\site-packages\ipykernel\kernelbase.py", line 510, in dispatch_queue
        await self.process_one()
      File "C:\Users\mttvi\anaconda3\lib\site-packages\ipykernel\kernelbase.py", line 499, in process_one
        await dispatch(*args)
      File "C:\Users\mttvi\anaconda3\lib\site-packages\ipykernel\kernelbase.py", line 406, in dispatch_shell
        await result
      File "C:\Users\mttvi\anaconda3\lib\site-packages\ipykernel\kernelbase.py", line 730, in execute_request
        reply_content = await reply_content
      File "C:\Users\mttvi\anaconda3\lib\site-packages\ipykernel\ipkernel.py", line 390, in do_execute
        res = shell.run_cell(code, store_history=store_history, silent=silent)
      File "C:\Users\mttvi\anaconda3\lib\site-packages\ipykernel\zmqshell.py", line 528, in run_cell
        return super().run_cell(*args, **kwargs)
      File "C:\Users\mttvi\anaconda3\lib\site-packages\IPython\core\interactiveshell.py", line 2914, in run_cell
        result = self._run_cell(
      File "C:\Users\mttvi\anaconda3\lib\site-packages\IPython\core\interactiveshell.py", line 2960, in _run_cell
        return runner(coro)
      File "C:\Users\mttvi\anaconda3\lib\site-packages\IPython\core\async_helpers.py", line 78, in _pseudo_sync_runner
        coro.send(None)
      File "C:\Users\mttvi\anaconda3\lib\site-packages\IPython\core\interactiveshell.py", line 3185, in run_cell_async
        has_raised = await self.run_ast_nodes(code_ast.body, cell_name,
      File "C:\Users\mttvi\anaconda3\lib\site-packages\IPython\core\interactiveshell.py", line 3377, in run_ast_nodes
        if (await self.run_code(code, result,  async_=asy)):
      File "C:\Users\mttvi\anaconda3\lib\site-packages\IPython\core\interactiveshell.py", line 3457, in run_code
        exec(code_obj, self.user_global_ns, self.user_ns)
      File "C:\Users\mttvi\AppData\Local\Temp\ipykernel_15100\3706768229.py", line 1, in <module>
        from txtai.embeddings import Embeddings
      File "C:\Users\mttvi\anaconda3\lib\site-packages\txtai\__init__.py", line 8, in <module>
        from .app import Application
      File "C:\Users\mttvi\anaconda3\lib\site-packages\txtai\app\__init__.py", line 5, in <module>
        from .base import Application, ReadOnlyError
      File "C:\Users\mttvi\anaconda3\lib\site-packages\txtai\app\base.py", line 12, in <module>
        from ..embeddings import Documents, Embeddings
      File "C:\Users\mttvi\anaconda3\lib\site-packages\txtai\embeddings\__init__.py", line 5, in <module>
        from .base import Embeddings
      File "C:\Users\mttvi\anaconda3\lib\site-packages\txtai\embeddings\base.py", line 16, in <module>
        from ..graph import GraphFactory
      File "C:\Users\mttvi\anaconda3\lib\site-packages\txtai\graph\__init__.py", line 5, in <module>
        from .base import Graph
      File "C:\Users\mttvi\anaconda3\lib\site-packages\txtai\graph\base.py", line 7, in <module>
        from .topics import Topics
      File "C:\Users\mttvi\anaconda3\lib\site-packages\txtai\graph\topics.py", line 5, in <module>
        from ..pipeline import Tokenizer
      File "C:\Users\mttvi\anaconda3\lib\site-packages\txtai\pipeline\__init__.py", line 5, in <module>
        from .audio import *
      File "C:\Users\mttvi\anaconda3\lib\site-packages\txtai\pipeline\audio\__init__.py", line 9, in <module>
        from .texttoaudio import TextToAudio
      File "C:\Users\mttvi\anaconda3\lib\site-packages\txtai\pipeline\audio\texttoaudio.py", line 5, in <module>
        from ..hfpipeline import HFPipeline
      File "C:\Users\mttvi\anaconda3\lib\site-packages\txtai\pipeline\hfpipeline.py", line 7, in <module>
        from transformers import pipeline
      File "C:\Users\mttvi\anaconda3\lib\site-packages\transformers\utils\import_utils.py", line 1754, in __getattr__
        module = self._get_module(self._class_to_module[name])
      File "C:\Users\mttvi\anaconda3\lib\site-packages\transformers\utils\import_utils.py", line 1764, in _get_module
        return importlib.import_module("." + module_name, self.__name__)
      File "C:\Users\mttvi\anaconda3\lib\importlib\__init__.py", line 127, in import_module
        return _bootstrap._gcd_import(name[level:], package, level)
      File "C:\Users\mttvi\anaconda3\lib\site-packages\transformers\pipelines\__init__.py", line 26, in <module>
        from ..image_processing_utils import BaseImageProcessor
      File "C:\Users\mttvi\anaconda3\lib\site-packages\transformers\image_processing_utils.py", line 21, in <module>
        from .image_transforms import center_crop, normalize, rescale
      File "C:\Users\mttvi\anaconda3\lib\site-packages\transformers\image_transforms.py", line 49, in <module>
        import tensorflow as tf
      File "C:\Users\mttvi\anaconda3\lib\site-packages\tensorflow\__init__.py", line 37, in <module>
        from tensorflow.python.tools import module_util as _module_util
      File "C:\Users\mttvi\anaconda3\lib\site-packages\tensorflow\python\__init__.py", line 37, in <module>
        from tensorflow.python.eager import context
      File "C:\Users\mttvi\anaconda3\lib\site-packages\tensorflow\python\eager\context.py", line 34, in <module>
        from tensorflow.python.client import pywrap_tf_session
      File "C:\Users\mttvi\anaconda3\lib\site-packages\tensorflow\python\client\pywrap_tf_session.py", line 19, in <module>
        from tensorflow.python.client._pywrap_tf_session import *
    


    ---------------------------------------------------------------------------

    AttributeError                            Traceback (most recent call last)

    AttributeError: _ARRAY_API not found


    
    A module that was compiled using NumPy 1.x cannot be run in
    NumPy 2.0.2 as it may crash. To support both 1.x and 2.x
    versions of NumPy, modules must be compiled with NumPy 2.0.
    Some module may need to rebuild instead e.g. with 'pybind11>=2.12'.
    
    If you are a user of the module, the easiest solution will be to
    downgrade to 'numpy<2' or try to upgrade the affected module.
    We expect that some modules will need time to support NumPy 2.
    
    Traceback (most recent call last):  File "C:\Users\mttvi\anaconda3\lib\runpy.py", line 197, in _run_module_as_main
        return _run_code(code, main_globals, None,
      File "C:\Users\mttvi\anaconda3\lib\runpy.py", line 87, in _run_code
        exec(code, run_globals)
      File "C:\Users\mttvi\anaconda3\lib\site-packages\ipykernel_launcher.py", line 17, in <module>
        app.launch_new_instance()
      File "C:\Users\mttvi\anaconda3\lib\site-packages\traitlets\config\application.py", line 846, in launch_instance
        app.start()
      File "C:\Users\mttvi\anaconda3\lib\site-packages\ipykernel\kernelapp.py", line 712, in start
        self.io_loop.start()
      File "C:\Users\mttvi\anaconda3\lib\site-packages\tornado\platform\asyncio.py", line 199, in start
        self.asyncio_loop.run_forever()
      File "C:\Users\mttvi\anaconda3\lib\asyncio\base_events.py", line 601, in run_forever
        self._run_once()
      File "C:\Users\mttvi\anaconda3\lib\asyncio\base_events.py", line 1905, in _run_once
        handle._run()
      File "C:\Users\mttvi\anaconda3\lib\asyncio\events.py", line 80, in _run
        self._context.run(self._callback, *self._args)
      File "C:\Users\mttvi\anaconda3\lib\site-packages\ipykernel\kernelbase.py", line 510, in dispatch_queue
        await self.process_one()
      File "C:\Users\mttvi\anaconda3\lib\site-packages\ipykernel\kernelbase.py", line 499, in process_one
        await dispatch(*args)
      File "C:\Users\mttvi\anaconda3\lib\site-packages\ipykernel\kernelbase.py", line 406, in dispatch_shell
        await result
      File "C:\Users\mttvi\anaconda3\lib\site-packages\ipykernel\kernelbase.py", line 730, in execute_request
        reply_content = await reply_content
      File "C:\Users\mttvi\anaconda3\lib\site-packages\ipykernel\ipkernel.py", line 390, in do_execute
        res = shell.run_cell(code, store_history=store_history, silent=silent)
      File "C:\Users\mttvi\anaconda3\lib\site-packages\ipykernel\zmqshell.py", line 528, in run_cell
        return super().run_cell(*args, **kwargs)
      File "C:\Users\mttvi\anaconda3\lib\site-packages\IPython\core\interactiveshell.py", line 2914, in run_cell
        result = self._run_cell(
      File "C:\Users\mttvi\anaconda3\lib\site-packages\IPython\core\interactiveshell.py", line 2960, in _run_cell
        return runner(coro)
      File "C:\Users\mttvi\anaconda3\lib\site-packages\IPython\core\async_helpers.py", line 78, in _pseudo_sync_runner
        coro.send(None)
      File "C:\Users\mttvi\anaconda3\lib\site-packages\IPython\core\interactiveshell.py", line 3185, in run_cell_async
        has_raised = await self.run_ast_nodes(code_ast.body, cell_name,
      File "C:\Users\mttvi\anaconda3\lib\site-packages\IPython\core\interactiveshell.py", line 3377, in run_ast_nodes
        if (await self.run_code(code, result,  async_=asy)):
      File "C:\Users\mttvi\anaconda3\lib\site-packages\IPython\core\interactiveshell.py", line 3457, in run_code
        exec(code_obj, self.user_global_ns, self.user_ns)
      File "C:\Users\mttvi\AppData\Local\Temp\ipykernel_15100\3706768229.py", line 1, in <module>
        from txtai.embeddings import Embeddings
      File "C:\Users\mttvi\anaconda3\lib\site-packages\txtai\__init__.py", line 8, in <module>
        from .app import Application
      File "C:\Users\mttvi\anaconda3\lib\site-packages\txtai\app\__init__.py", line 5, in <module>
        from .base import Application, ReadOnlyError
      File "C:\Users\mttvi\anaconda3\lib\site-packages\txtai\app\base.py", line 12, in <module>
        from ..embeddings import Documents, Embeddings
      File "C:\Users\mttvi\anaconda3\lib\site-packages\txtai\embeddings\__init__.py", line 5, in <module>
        from .base import Embeddings
      File "C:\Users\mttvi\anaconda3\lib\site-packages\txtai\embeddings\base.py", line 16, in <module>
        from ..graph import GraphFactory
      File "C:\Users\mttvi\anaconda3\lib\site-packages\txtai\graph\__init__.py", line 5, in <module>
        from .base import Graph
      File "C:\Users\mttvi\anaconda3\lib\site-packages\txtai\graph\base.py", line 7, in <module>
        from .topics import Topics
      File "C:\Users\mttvi\anaconda3\lib\site-packages\txtai\graph\topics.py", line 5, in <module>
        from ..pipeline import Tokenizer
      File "C:\Users\mttvi\anaconda3\lib\site-packages\txtai\pipeline\__init__.py", line 5, in <module>
        from .audio import *
      File "C:\Users\mttvi\anaconda3\lib\site-packages\txtai\pipeline\audio\__init__.py", line 9, in <module>
        from .texttoaudio import TextToAudio
      File "C:\Users\mttvi\anaconda3\lib\site-packages\txtai\pipeline\audio\texttoaudio.py", line 5, in <module>
        from ..hfpipeline import HFPipeline
      File "C:\Users\mttvi\anaconda3\lib\site-packages\txtai\pipeline\hfpipeline.py", line 7, in <module>
        from transformers import pipeline
      File "C:\Users\mttvi\anaconda3\lib\site-packages\transformers\utils\import_utils.py", line 1754, in __getattr__
        module = self._get_module(self._class_to_module[name])
      File "C:\Users\mttvi\anaconda3\lib\site-packages\transformers\utils\import_utils.py", line 1764, in _get_module
        return importlib.import_module("." + module_name, self.__name__)
      File "C:\Users\mttvi\anaconda3\lib\importlib\__init__.py", line 127, in import_module
        return _bootstrap._gcd_import(name[level:], package, level)
      File "C:\Users\mttvi\anaconda3\lib\site-packages\transformers\pipelines\__init__.py", line 26, in <module>
        from ..image_processing_utils import BaseImageProcessor
      File "C:\Users\mttvi\anaconda3\lib\site-packages\transformers\image_processing_utils.py", line 21, in <module>
        from .image_transforms import center_crop, normalize, rescale
      File "C:\Users\mttvi\anaconda3\lib\site-packages\transformers\image_transforms.py", line 49, in <module>
        import tensorflow as tf
      File "C:\Users\mttvi\anaconda3\lib\site-packages\tensorflow\__init__.py", line 37, in <module>
        from tensorflow.python.tools import module_util as _module_util
      File "C:\Users\mttvi\anaconda3\lib\site-packages\tensorflow\python\__init__.py", line 42, in <module>
        from tensorflow.python import data
      File "C:\Users\mttvi\anaconda3\lib\site-packages\tensorflow\python\data\__init__.py", line 21, in <module>
        from tensorflow.python.data import experimental
      File "C:\Users\mttvi\anaconda3\lib\site-packages\tensorflow\python\data\experimental\__init__.py", line 96, in <module>
        from tensorflow.python.data.experimental import service
      File "C:\Users\mttvi\anaconda3\lib\site-packages\tensorflow\python\data\experimental\service\__init__.py", line 419, in <module>
        from tensorflow.python.data.experimental.ops.data_service_ops import distribute
      File "C:\Users\mttvi\anaconda3\lib\site-packages\tensorflow\python\data\experimental\ops\data_service_ops.py", line 22, in <module>
        from tensorflow.python.data.experimental.ops import compression_ops
      File "C:\Users\mttvi\anaconda3\lib\site-packages\tensorflow\python\data\experimental\ops\compression_ops.py", line 16, in <module>
        from tensorflow.python.data.util import structure
      File "C:\Users\mttvi\anaconda3\lib\site-packages\tensorflow\python\data\util\structure.py", line 22, in <module>
        from tensorflow.python.data.util import nest
      File "C:\Users\mttvi\anaconda3\lib\site-packages\tensorflow\python\data\util\nest.py", line 34, in <module>
        from tensorflow.python.framework import sparse_tensor as _sparse_tensor
      File "C:\Users\mttvi\anaconda3\lib\site-packages\tensorflow\python\framework\sparse_tensor.py", line 24, in <module>
        from tensorflow.python.framework import constant_op
      File "C:\Users\mttvi\anaconda3\lib\site-packages\tensorflow\python\framework\constant_op.py", line 25, in <module>
        from tensorflow.python.eager import execute
      File "C:\Users\mttvi\anaconda3\lib\site-packages\tensorflow\python\eager\execute.py", line 21, in <module>
        from tensorflow.python.framework import dtypes
      File "C:\Users\mttvi\anaconda3\lib\site-packages\tensorflow\python\framework\dtypes.py", line 29, in <module>
        from tensorflow.python.lib.core import _pywrap_bfloat16
    


    ---------------------------------------------------------------------------

    AttributeError                            Traceback (most recent call last)

    AttributeError: _ARRAY_API not found



    ---------------------------------------------------------------------------

    ImportError                               Traceback (most recent call last)

    ImportError: numpy.core._multiarray_umath failed to import



    ---------------------------------------------------------------------------

    ImportError                               Traceback (most recent call last)

    ImportError: numpy.core.umath failed to import



    ---------------------------------------------------------------------------

    TypeError                                 Traceback (most recent call last)

    ~\anaconda3\lib\site-packages\transformers\utils\import_utils.py in _get_module(self, module_name)
       1763         try:
    -> 1764             return importlib.import_module("." + module_name, self.__name__)
       1765         except Exception as e:
    

    ~\anaconda3\lib\importlib\__init__.py in import_module(name, package)
        126             level += 1
    --> 127     return _bootstrap._gcd_import(name[level:], package, level)
        128 
    

    ~\anaconda3\lib\importlib\_bootstrap.py in _gcd_import(name, package, level)
    

    ~\anaconda3\lib\importlib\_bootstrap.py in _find_and_load(name, import_)
    

    ~\anaconda3\lib\importlib\_bootstrap.py in _find_and_load_unlocked(name, import_)
    

    ~\anaconda3\lib\importlib\_bootstrap.py in _load_unlocked(spec)
    

    ~\anaconda3\lib\importlib\_bootstrap_external.py in exec_module(self, module)
    

    ~\anaconda3\lib\importlib\_bootstrap.py in _call_with_frames_removed(f, *args, **kwds)
    

    ~\anaconda3\lib\site-packages\transformers\pipelines\__init__.py in <module>
         25 from ..feature_extraction_utils import PreTrainedFeatureExtractor
    ---> 26 from ..image_processing_utils import BaseImageProcessor
         27 from ..models.auto.configuration_auto import AutoConfig
    

    ~\anaconda3\lib\site-packages\transformers\image_processing_utils.py in <module>
         20 from .image_processing_base import BatchFeature, ImageProcessingMixin
    ---> 21 from .image_transforms import center_crop, normalize, rescale
         22 from .image_utils import ChannelDimension
    

    ~\anaconda3\lib\site-packages\transformers\image_transforms.py in <module>
         48 if is_tf_available():
    ---> 49     import tensorflow as tf
         50 
    

    ~\anaconda3\lib\site-packages\tensorflow\__init__.py in <module>
         36 
    ---> 37 from tensorflow.python.tools import module_util as _module_util
         38 from tensorflow.python.util.lazy_loader import LazyLoader as _LazyLoader
    

    ~\anaconda3\lib\site-packages\tensorflow\python\__init__.py in <module>
         41 # Bring in subpackages.
    ---> 42 from tensorflow.python import data
         43 from tensorflow.python import distribute
    

    ~\anaconda3\lib\site-packages\tensorflow\python\data\__init__.py in <module>
         20 # pylint: disable=unused-import
    ---> 21 from tensorflow.python.data import experimental
         22 from tensorflow.python.data.ops.dataset_ops import AUTOTUNE
    

    ~\anaconda3\lib\site-packages\tensorflow\python\data\experimental\__init__.py in <module>
         95 # pylint: disable=unused-import
    ---> 96 from tensorflow.python.data.experimental import service
         97 from tensorflow.python.data.experimental.ops.batching import dense_to_ragged_batch
    

    ~\anaconda3\lib\site-packages\tensorflow\python\data\experimental\service\__init__.py in <module>
        418 
    --> 419 from tensorflow.python.data.experimental.ops.data_service_ops import distribute
        420 from tensorflow.python.data.experimental.ops.data_service_ops import from_dataset_id
    

    ~\anaconda3\lib\site-packages\tensorflow\python\data\experimental\ops\data_service_ops.py in <module>
         21 from tensorflow.python import tf2
    ---> 22 from tensorflow.python.data.experimental.ops import compression_ops
         23 from tensorflow.python.data.experimental.service import _pywrap_server_lib
    

    ~\anaconda3\lib\site-packages\tensorflow\python\data\experimental\ops\compression_ops.py in <module>
         15 """Ops for compressing and uncompressing dataset elements."""
    ---> 16 from tensorflow.python.data.util import structure
         17 from tensorflow.python.ops import gen_experimental_dataset_ops as ged_ops
    

    ~\anaconda3\lib\site-packages\tensorflow\python\data\util\structure.py in <module>
         21 
    ---> 22 from tensorflow.python.data.util import nest
         23 from tensorflow.python.framework import composite_tensor
    

    ~\anaconda3\lib\site-packages\tensorflow\python\data\util\nest.py in <module>
         33 
    ---> 34 from tensorflow.python.framework import sparse_tensor as _sparse_tensor
         35 from tensorflow.python.util import _pywrap_utils
    

    ~\anaconda3\lib\site-packages\tensorflow\python\framework\sparse_tensor.py in <module>
         23 from tensorflow.python.framework import composite_tensor
    ---> 24 from tensorflow.python.framework import constant_op
         25 from tensorflow.python.framework import dtypes
    

    ~\anaconda3\lib\site-packages\tensorflow\python\framework\constant_op.py in <module>
         24 from tensorflow.python.eager import context
    ---> 25 from tensorflow.python.eager import execute
         26 from tensorflow.python.framework import dtypes
    

    ~\anaconda3\lib\site-packages\tensorflow\python\eager\execute.py in <module>
         20 from tensorflow.python.eager import core
    ---> 21 from tensorflow.python.framework import dtypes
         22 from tensorflow.python.framework import ops
    

    ~\anaconda3\lib\site-packages\tensorflow\python\framework\dtypes.py in <module>
         33 
    ---> 34 _np_bfloat16 = _pywrap_bfloat16.TF_bfloat16_type()
         35 
    

    TypeError: Unable to convert function return value to a Python type! The signature was
    	() -> handle

    
    The above exception was the direct cause of the following exception:
    

    RuntimeError                              Traceback (most recent call last)

    ~\AppData\Local\Temp\ipykernel_15100\3706768229.py in <module>
    ----> 1 from txtai.embeddings import Embeddings
          2 
          3 embeddings = Embeddings()
          4 embeddings.index([("1", "Text data to index")])
    

    ~\anaconda3\lib\site-packages\txtai\__init__.py in <module>
          6 
          7 # Top-level imports
    ----> 8 from .app import Application
          9 from .embeddings import Embeddings
         10 from .pipeline import LLM, RAG
    

    ~\anaconda3\lib\site-packages\txtai\app\__init__.py in <module>
          3 """
          4 
    ----> 5 from .base import Application, ReadOnlyError
    

    ~\anaconda3\lib\site-packages\txtai\app\base.py in <module>
         10 import yaml
         11 
    ---> 12 from ..embeddings import Documents, Embeddings
         13 from ..pipeline import PipelineFactory
         14 from ..workflow import WorkflowFactory
    

    ~\anaconda3\lib\site-packages\txtai\embeddings\__init__.py in <module>
          3 """
          4 
    ----> 5 from .base import Embeddings
          6 from .index import *
          7 from .search import *
    

    ~\anaconda3\lib\site-packages\txtai\embeddings\base.py in <module>
         14 from ..cloud import CloudFactory
         15 from ..database import DatabaseFactory
    ---> 16 from ..graph import GraphFactory
         17 from ..scoring import ScoringFactory
         18 from ..vectors import VectorsFactory
    

    ~\anaconda3\lib\site-packages\txtai\graph\__init__.py in <module>
          3 """
          4 
    ----> 5 from .base import Graph
          6 from .factory import GraphFactory
          7 from .networkx import NetworkX
    

    ~\anaconda3\lib\site-packages\txtai\graph\base.py in <module>
          5 from collections import Counter
          6 
    ----> 7 from .topics import Topics
          8 
          9 
    

    ~\anaconda3\lib\site-packages\txtai\graph\topics.py in <module>
          3 """
          4 
    ----> 5 from ..pipeline import Tokenizer
          6 from ..scoring import ScoringFactory
          7 
    

    ~\anaconda3\lib\site-packages\txtai\pipeline\__init__.py in <module>
          3 """
          4 
    ----> 5 from .audio import *
          6 from .base import Pipeline
          7 from .data import *
    

    ~\anaconda3\lib\site-packages\txtai\pipeline\audio\__init__.py in <module>
          7 from .microphone import Microphone
          8 from .signal import Signal
    ----> 9 from .texttoaudio import TextToAudio
         10 from .texttospeech import TextToSpeech
         11 from .transcription import Transcription
    

    ~\anaconda3\lib\site-packages\txtai\pipeline\audio\texttoaudio.py in <module>
          3 """
          4 
    ----> 5 from ..hfpipeline import HFPipeline
          6 from .signal import Signal, SCIPY
          7 
    

    ~\anaconda3\lib\site-packages\txtai\pipeline\hfpipeline.py in <module>
          5 import inspect
          6 
    ----> 7 from transformers import pipeline
          8 
          9 from ..models import Models
    

    ~\anaconda3\lib\importlib\_bootstrap.py in _handle_fromlist(module, fromlist, import_, recursive)
    

    ~\anaconda3\lib\site-packages\transformers\utils\import_utils.py in __getattr__(self, name)
       1752             value = Placeholder
       1753         elif name in self._class_to_module.keys():
    -> 1754             module = self._get_module(self._class_to_module[name])
       1755             value = getattr(module, name)
       1756         else:
    

    ~\anaconda3\lib\site-packages\transformers\utils\import_utils.py in _get_module(self, module_name)
       1764             return importlib.import_module("." + module_name, self.__name__)
       1765         except Exception as e:
    -> 1766             raise RuntimeError(
       1767                 f"Failed to import {self.__name__}.{module_name} because of the following error (look up to see its"
       1768                 f" traceback):\n{e}"
    

    RuntimeError: Failed to import transformers.pipelines because of the following error (look up to see its traceback):
    Unable to convert function return value to a Python type! The signature was
    	() -> handle


####  Funcionalidade para Definir os "Chunks":


```python
pip install langchain
```

    Collecting langchain
      Using cached langchain-0.3.3-py3-none-any.whl (1.0 MB)
    Requirement already satisfied: PyYAML>=5.3 in c:\users\mttvi\anaconda3\lib\site-packages (from langchain) (6.0)
    Collecting aiohttp<4.0.0,>=3.8.3
      Using cached aiohttp-3.10.10-cp39-cp39-win_amd64.whl (381 kB)
    Requirement already satisfied: tenacity!=8.4.0,<9.0.0,>=8.1.0 in c:\users\mttvi\anaconda3\lib\site-packages (from langchain) (8.2.3)
    Requirement already satisfied: SQLAlchemy<3,>=1.4 in c:\users\mttvi\anaconda3\lib\site-packages (from langchain) (1.4.39)
    Requirement already satisfied: requests<3,>=2 in c:\users\mttvi\anaconda3\lib\site-packages (from langchain) (2.28.1)
    Collecting langchain-text-splitters<0.4.0,>=0.3.0
      Using cached langchain_text_splitters-0.3.0-py3-none-any.whl (25 kB)
    Collecting async-timeout<5.0.0,>=4.0.0
      Using cached async_timeout-4.0.3-py3-none-any.whl (5.7 kB)
    Requirement already satisfied: numpy<2,>=1 in c:\users\mttvi\anaconda3\lib\site-packages (from langchain) (1.26.4)
    Collecting langchain-core<0.4.0,>=0.3.10
      Using cached langchain_core-0.3.10-py3-none-any.whl (404 kB)
    Collecting langsmith<0.2.0,>=0.1.17
      Using cached langsmith-0.1.135-py3-none-any.whl (295 kB)
    Collecting pydantic<3.0.0,>=2.7.4
      Using cached pydantic-2.9.2-py3-none-any.whl (434 kB)
    Collecting multidict<7.0,>=4.5
      Using cached multidict-6.1.0-cp39-cp39-win_amd64.whl (28 kB)
    Collecting yarl<2.0,>=1.12.0
      Using cached yarl-1.15.2-cp39-cp39-win_amd64.whl (84 kB)
    Requirement already satisfied: attrs>=17.3.0 in c:\users\mttvi\anaconda3\lib\site-packages (from aiohttp<4.0.0,>=3.8.3->langchain) (21.4.0)
    Collecting aiohappyeyeballs>=2.3.0
      Using cached aiohappyeyeballs-2.4.3-py3-none-any.whl (14 kB)
    Collecting frozenlist>=1.1.1
      Using cached frozenlist-1.4.1-cp39-cp39-win_amd64.whl (50 kB)
    Collecting aiosignal>=1.1.2
      Using cached aiosignal-1.3.1-py3-none-any.whl (7.6 kB)
    Requirement already satisfied: typing-extensions>=4.7 in c:\users\mttvi\anaconda3\lib\site-packages (from langchain-core<0.4.0,>=0.3.10->langchain) (4.12.2)
    Collecting jsonpatch<2.0,>=1.33
      Using cached jsonpatch-1.33-py2.py3-none-any.whl (12 kB)
    Requirement already satisfied: packaging<25,>=23.2 in c:\users\mttvi\anaconda3\lib\site-packages (from langchain-core<0.4.0,>=0.3.10->langchain) (24.1)
    Requirement already satisfied: orjson<4.0.0,>=3.9.14 in c:\users\mttvi\anaconda3\lib\site-packages (from langsmith<0.2.0,>=0.1.17->langchain) (3.10.7)
    Collecting httpx<1,>=0.23.0
      Using cached httpx-0.27.2-py3-none-any.whl (76 kB)
    Collecting requests-toolbelt<2.0.0,>=1.0.0
      Using cached requests_toolbelt-1.0.0-py2.py3-none-any.whl (54 kB)
    Requirement already satisfied: pydantic-core==2.23.4 in c:\users\mttvi\anaconda3\lib\site-packages (from pydantic<3.0.0,>=2.7.4->langchain) (2.23.4)
    Collecting annotated-types>=0.6.0
      Using cached annotated_types-0.7.0-py3-none-any.whl (13 kB)
    Requirement already satisfied: charset-normalizer<3,>=2 in c:\users\mttvi\anaconda3\lib\site-packages (from requests<3,>=2->langchain) (2.0.4)
    Requirement already satisfied: idna<4,>=2.5 in c:\users\mttvi\anaconda3\lib\site-packages (from requests<3,>=2->langchain) (3.3)
    Requirement already satisfied: urllib3<1.27,>=1.21.1 in c:\users\mttvi\anaconda3\lib\site-packages (from requests<3,>=2->langchain) (1.26.11)
    Requirement already satisfied: certifi>=2017.4.17 in c:\users\mttvi\anaconda3\lib\site-packages (from requests<3,>=2->langchain) (2022.9.14)
    Requirement already satisfied: greenlet!=0.4.17 in c:\users\mttvi\anaconda3\lib\site-packages (from SQLAlchemy<3,>=1.4->langchain) (1.1.1)
    Collecting httpcore==1.*
      Using cached httpcore-1.0.6-py3-none-any.whl (78 kB)
    Requirement already satisfied: anyio in c:\users\mttvi\anaconda3\lib\site-packages (from httpx<1,>=0.23.0->langsmith<0.2.0,>=0.1.17->langchain) (3.5.0)
    Requirement already satisfied: sniffio in c:\users\mttvi\anaconda3\lib\site-packages (from httpx<1,>=0.23.0->langsmith<0.2.0,>=0.1.17->langchain) (1.2.0)
    Requirement already satisfied: h11<0.15,>=0.13 in c:\users\mttvi\anaconda3\lib\site-packages (from httpcore==1.*->httpx<1,>=0.23.0->langsmith<0.2.0,>=0.1.17->langchain) (0.14.0)
    Collecting jsonpointer>=1.9
      Using cached jsonpointer-3.0.0-py2.py3-none-any.whl (7.6 kB)
    Requirement already satisfied: propcache>=0.2.0 in c:\users\mttvi\anaconda3\lib\site-packages (from yarl<2.0,>=1.12.0->aiohttp<4.0.0,>=3.8.3->langchain) (0.2.0)
    Installing collected packages: multidict, jsonpointer, httpcore, frozenlist, async-timeout, annotated-types, aiohappyeyeballs, yarl, requests-toolbelt, pydantic, jsonpatch, httpx, aiosignal, langsmith, aiohttp, langchain-core, langchain-text-splitters, langchain
    Note: you may need to restart the kernel to use updated packages.
    

    WARNING: Ignoring invalid distribution -otobuf (c:\users\mttvi\anaconda3\lib\site-packages)
    WARNING: Ignoring invalid distribution -rotobuf (c:\users\mttvi\anaconda3\lib\site-packages)
    WARNING: Ignoring invalid distribution - (c:\users\mttvi\anaconda3\lib\site-packages)
    WARNING: Ignoring invalid distribution -otobuf (c:\users\mttvi\anaconda3\lib\site-packages)
    WARNING: Ignoring invalid distribution -rotobuf (c:\users\mttvi\anaconda3\lib\site-packages)
    WARNING: Ignoring invalid distribution - (c:\users\mttvi\anaconda3\lib\site-packages)
    WARNING: Ignoring invalid distribution -otobuf (c:\users\mttvi\anaconda3\lib\site-packages)
    WARNING: Ignoring invalid distribution -rotobuf (c:\users\mttvi\anaconda3\lib\site-packages)
    WARNING: Ignoring invalid distribution - (c:\users\mttvi\anaconda3\lib\site-packages)
    ERROR: Could not install packages due to an OSError: [WinError 32] O arquivo já está sendo usado por outro processo: 'C:\\Users\\mttvi\\anaconda3\\Lib\\site-packages\\langsmith\\evaluation\\integrations\\test.excalidraw.png'
    Consider using the `--user` option or check the permissions.
    
    WARNING: Ignoring invalid distribution -otobuf (c:\users\mttvi\anaconda3\lib\site-packages)
    WARNING: Ignoring invalid distribution -rotobuf (c:\users\mttvi\anaconda3\lib\site-packages)
    WARNING: Ignoring invalid distribution - (c:\users\mttvi\anaconda3\lib\site-packages)
    WARNING: Ignoring invalid distribution -otobuf (c:\users\mttvi\anaconda3\lib\site-packages)
    WARNING: Ignoring invalid distribution -rotobuf (c:\users\mttvi\anaconda3\lib\site-packages)
    WARNING: Ignoring invalid distribution - (c:\users\mttvi\anaconda3\lib\site-packages)
    WARNING: Ignoring invalid distribution -otobuf (c:\users\mttvi\anaconda3\lib\site-packages)
    WARNING: Ignoring invalid distribution -rotobuf (c:\users\mttvi\anaconda3\lib\site-packages)
    WARNING: Ignoring invalid distribution - (c:\users\mttvi\anaconda3\lib\site-packages)
    

    Files removed: 740
    


```python
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Defina o texto que será dividido
text = """
Página 1
A Bosch é líder mundial no fornecimento de tecnologia e serviços. A empresa emprega
aproximadamente 400.000 colaboradores em todo o mundo. Neste relatório, apresentamos um resumo 
das atividades de atendimento ao cliente, vendas, reclamações e elogios, bem como gráficos e análises 
estatísticas dos dados. Este documento contém informações detalhadas sobre o desempenho da empresa 
durante os últimos 12 meses. Inclui métricas de desempenho e insights valiosos para entender melhor as 
tendências do mercado.

Informação adicional: A Bosch é conhecida por sua inovação e qualidade. Nossa equipe está comprometida 
em fornecer as melhores soluções tecnológicas para nossos clientes. Estamos sempre à frente das tendências 
do mercado, garantindo a satisfação e confiança de nossos consumidores.

Página 2
A Bosch é líder mundial no fornecimento de tecnologia e serviços. A empresa emprega aproximadamente 400.000
"""

# Configura o text_splitter com tamanho de chunk e sobreposição
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)

# Divide o texto em chunks
chunks = text_splitter.split_text(text)

# Exibe os chunks divididos
for i, chunk in enumerate(chunks):
    print(f"Chunk {i+1}:\n{chunk}\n")

```

    Chunk 1:
    Página 1
    A Bosch é líder mundial no fornecimento de tecnologia e serviços. A empresa emprega
    aproximadamente 400.000 colaboradores em todo o mundo. Neste relatório, apresentamos um resumo 
    das atividades de atendimento ao cliente, vendas, reclamações e elogios, bem como gráficos e análises 
    estatísticas dos dados. Este documento contém informações detalhadas sobre o desempenho da empresa 
    durante os últimos 12 meses. Inclui métricas de desempenho e insights valiosos para entender melhor as
    
    Chunk 2:
    tendências do mercado.
    
    Chunk 3:
    Informação adicional: A Bosch é conhecida por sua inovação e qualidade. Nossa equipe está comprometida 
    em fornecer as melhores soluções tecnológicas para nossos clientes. Estamos sempre à frente das tendências 
    do mercado, garantindo a satisfação e confiança de nossos consumidores.
    
    Página 2
    A Bosch é líder mundial no fornecimento de tecnologia e serviços. A empresa emprega aproximadamente 400.000
    
    

####  Métrica de Avaliação da Qualidade:


```python
!pip uninstall numpy scikit-learn
```

    WARNING: Ignoring invalid distribution -otobuf (c:\users\mttvi\anaconda3\lib\site-packages)
    WARNING: Ignoring invalid distribution -rotobuf (c:\users\mttvi\anaconda3\lib\site-packages)
    WARNING: Ignoring invalid distribution - (c:\users\mttvi\anaconda3\lib\site-packages)
    WARNING: Ignoring invalid distribution -otobuf (c:\users\mttvi\anaconda3\lib\site-packages)
    WARNING: Ignoring invalid distribution -rotobuf (c:\users\mttvi\anaconda3\lib\site-packages)
    WARNING: Ignoring invalid distribution - (c:\users\mttvi\anaconda3\lib\site-packages)
    WARNING: Ignoring invalid distribution -otobuf (c:\users\mttvi\anaconda3\lib\site-packages)
    WARNING: Ignoring invalid distribution -rotobuf (c:\users\mttvi\anaconda3\lib\site-packages)
    WARNING: Ignoring invalid distribution - (c:\users\mttvi\anaconda3\lib\site-packages)
        WARNING: Ignoring invalid distribution -otobuf (c:\users\mttvi\anaconda3\lib\site-packages)
        WARNING: Ignoring invalid distribution -rotobuf (c:\users\mttvi\anaconda3\lib\site-packages)
        WARNING: Ignoring invalid distribution - (c:\users\mttvi\anaconda3\lib\site-packages)
        WARNING: Ignoring invalid distribution -otobuf (c:\users\mttvi\anaconda3\lib\site-packages)
        WARNING: Ignoring invalid distribution -rotobuf (c:\users\mttvi\anaconda3\lib\site-packages)
        WARNING: Ignoring invalid distribution - (c:\users\mttvi\anaconda3\lib\site-packages)
    ERROR: Could not install packages due to an OSError: [WinError 5] Acesso negado: 'C:\\Users\\mttvi\\anaconda3\\Lib\\site-packages\\~cipy.libs\\libopenblas_v0.3.27--3aa239bc726cfb0bd8e5330d8d4c15c6.dll'
    Consider using the `--user` option or check the permissions.
    
    WARNING: Ignoring invalid distribution -otobuf (c:\users\mttvi\anaconda3\lib\site-packages)
    WARNING: Ignoring invalid distribution -rotobuf (c:\users\mttvi\anaconda3\lib\site-packages)
    WARNING: Ignoring invalid distribution - (c:\users\mttvi\anaconda3\lib\site-packages)
    WARNING: Ignoring invalid distribution -otobuf (c:\users\mttvi\anaconda3\lib\site-packages)
    WARNING: Ignoring invalid distribution -rotobuf (c:\users\mttvi\anaconda3\lib\site-packages)
    WARNING: Ignoring invalid distribution - (c:\users\mttvi\anaconda3\lib\site-packages)
    WARNING: Ignoring invalid distribution -otobuf (c:\users\mttvi\anaconda3\lib\site-packages)
    WARNING: Ignoring invalid distribution -rotobuf (c:\users\mttvi\anaconda3\lib\site-packages)
    WARNING: Ignoring invalid distribution - (c:\users\mttvi\anaconda3\lib\site-packages)
    

    Collecting numpy==1.21.6
      Downloading numpy-1.21.6-cp39-cp39-win_amd64.whl (14.0 MB)
         ---------------------------------------- 14.0/14.0 MB 1.6 MB/s eta 0:00:00
    Requirement already satisfied: scikit-learn==1.0.2 in c:\users\mttvi\anaconda3\lib\site-packages (1.0.2)
    Requirement already satisfied: threadpoolctl>=2.0.0 in c:\users\mttvi\anaconda3\lib\site-packages (from scikit-learn==1.0.2) (2.2.0)
    Requirement already satisfied: joblib>=0.11 in c:\users\mttvi\anaconda3\lib\site-packages (from scikit-learn==1.0.2) (1.1.0)
    Requirement already satisfied: scipy>=1.1.0 in c:\users\mttvi\anaconda3\lib\site-packages (from scikit-learn==1.0.2) (1.13.1)
    Collecting scipy>=1.1.0
      Downloading scipy-1.13.0-cp39-cp39-win_amd64.whl (46.2 MB)
         ---------------------------------------- 46.2/46.2 MB 1.0 MB/s eta 0:00:00
      Downloading scipy-1.12.0-cp39-cp39-win_amd64.whl (46.2 MB)
         ---------------------------------------- 46.2/46.2 MB 4.0 MB/s eta 0:00:00
      Downloading scipy-1.11.4-cp39-cp39-win_amd64.whl (44.3 MB)
         ---------------------------------------- 44.3/44.3 MB 2.8 MB/s eta 0:00:00
    Installing collected packages: numpy, scipy
      Attempting uninstall: numpy
        Found existing installation: numpy 1.26.4
        Uninstalling numpy-1.26.4:
          Successfully uninstalled numpy-1.26.4
      Attempting uninstall: scipy
        Found existing installation: scipy 1.13.1
        Uninstalling scipy-1.13.1:
          Successfully uninstalled scipy-1.13.1
    ^C
    


```python
!pip install numpy scikit-learn
```

    Requirement already satisfied: numpy in c:\users\mttvi\anaconda3\lib\site-packages (1.21.6)

    WARNING: Ignoring invalid distribution -otobuf (c:\users\mttvi\anaconda3\lib\site-packages)
    WARNING: Ignoring invalid distribution -rotobuf (c:\users\mttvi\anaconda3\lib\site-packages)
    WARNING: Ignoring invalid distribution - (c:\users\mttvi\anaconda3\lib\site-packages)
    WARNING: Ignoring invalid distribution -otobuf (c:\users\mttvi\anaconda3\lib\site-packages)
    WARNING: Ignoring invalid distribution -rotobuf (c:\users\mttvi\anaconda3\lib\site-packages)
    WARNING: Ignoring invalid distribution - (c:\users\mttvi\anaconda3\lib\site-packages)
    WARNING: Ignoring invalid distribution -otobuf (c:\users\mttvi\anaconda3\lib\site-packages)
    WARNING: Ignoring invalid distribution -rotobuf (c:\users\mttvi\anaconda3\lib\site-packages)
    WARNING: Ignoring invalid distribution - (c:\users\mttvi\anaconda3\lib\site-packages)
    WARNING: Ignoring invalid distribution -otobuf (c:\users\mttvi\anaconda3\lib\site-packages)
    WARNING: Ignoring invalid distribution -rotobuf (c:\users\mttvi\anaconda3\lib\site-packages)
    WARNING: Ignoring invalid distribution - (c:\users\mttvi\anaconda3\lib\site-packages)
    WARNING: Ignoring invalid distribution -otobuf (c:\users\mttvi\anaconda3\lib\site-packages)
    WARNING: Ignoring invalid distribution -rotobuf (c:\users\mttvi\anaconda3\lib\site-packages)
    WARNING: Ignoring invalid distribution - (c:\users\mttvi\anaconda3\lib\site-packages)
    WARNING: Ignoring invalid distribution -otobuf (c:\users\mttvi\anaconda3\lib\site-packages)
    WARNING: Ignoring invalid distribution -rotobuf (c:\users\mttvi\anaconda3\lib\site-packages)
    WARNING: Ignoring invalid distribution - (c:\users\mttvi\anaconda3\lib\site-packages)
    

    
    Requirement already satisfied: scikit-learn in c:\users\mttvi\anaconda3\lib\site-packages (1.0.2)
    Requirement already satisfied: joblib>=0.11 in c:\users\mttvi\anaconda3\lib\site-packages (from scikit-learn) (1.1.0)
    Requirement already satisfied: threadpoolctl>=2.0.0 in c:\users\mttvi\anaconda3\lib\site-packages (from scikit-learn) (2.2.0)
    Requirement already satisfied: scipy>=1.1.0 in c:\users\mttvi\anaconda3\lib\site-packages (from scikit-learn) (1.11.4)
    


```python
!pip install numpy==1.21.6 scikit-learn==1.0.2
```

    Requirement already satisfied: numpy==1.21.6 in c:\users\mttvi\anaconda3\lib\site-packages (1.21.6)
    Requirement already satisfied: scikit-learn==1.0.2 in c:\users\mttvi\anaconda3\lib\site-packages (1.0.2)
    Requirement already satisfied: threadpoolctl>=2.0.0 in c:\users\mttvi\anaconda3\lib\site-packages (from scikit-learn==1.0.2) (2.2.0)
    Requirement already satisfied: scipy>=1.1.0 in c:\users\mttvi\anaconda3\lib\site-packages (from scikit-learn==1.0.2) (1.11.4)
    Requirement already satisfied: joblib>=0.11 in c:\users\mttvi\anaconda3\lib\site-packages (from scikit-learn==1.0.2) (1.1.0)
    

    WARNING: Ignoring invalid distribution -otobuf (c:\users\mttvi\anaconda3\lib\site-packages)
    WARNING: Ignoring invalid distribution -rotobuf (c:\users\mttvi\anaconda3\lib\site-packages)
    WARNING: Ignoring invalid distribution - (c:\users\mttvi\anaconda3\lib\site-packages)
    WARNING: Ignoring invalid distribution -otobuf (c:\users\mttvi\anaconda3\lib\site-packages)
    WARNING: Ignoring invalid distribution -rotobuf (c:\users\mttvi\anaconda3\lib\site-packages)
    WARNING: Ignoring invalid distribution - (c:\users\mttvi\anaconda3\lib\site-packages)
    WARNING: Ignoring invalid distribution -otobuf (c:\users\mttvi\anaconda3\lib\site-packages)
    WARNING: Ignoring invalid distribution -rotobuf (c:\users\mttvi\anaconda3\lib\site-packages)
    WARNING: Ignoring invalid distribution - (c:\users\mttvi\anaconda3\lib\site-packages)
    WARNING: Ignoring invalid distribution -otobuf (c:\users\mttvi\anaconda3\lib\site-packages)
    WARNING: Ignoring invalid distribution -rotobuf (c:\users\mttvi\anaconda3\lib\site-packages)
    WARNING: Ignoring invalid distribution - (c:\users\mttvi\anaconda3\lib\site-packages)
    WARNING: Ignoring invalid distribution -otobuf (c:\users\mttvi\anaconda3\lib\site-packages)
    WARNING: Ignoring invalid distribution -rotobuf (c:\users\mttvi\anaconda3\lib\site-packages)
    WARNING: Ignoring invalid distribution - (c:\users\mttvi\anaconda3\lib\site-packages)
    WARNING: Ignoring invalid distribution -otobuf (c:\users\mttvi\anaconda3\lib\site-packages)
    WARNING: Ignoring invalid distribution -rotobuf (c:\users\mttvi\anaconda3\lib\site-packages)
    WARNING: Ignoring invalid distribution - (c:\users\mttvi\anaconda3\lib\site-packages)
    


```python
!pip cache purge
```

    Files removed: 4
    


```python
pip install --upgrade numpy scikit-learn
```

    Requirement already satisfied: numpy in c:\users\mttvi\anaconda3\lib\site-packages (1.21.6)
    Collecting numpy
      Downloading numpy-2.0.2-cp39-cp39-win_amd64.whl (15.9 MB)
         -------------------------------------- 15.9/15.9 MB 652.0 kB/s eta 0:00:00
    Requirement already satisfied: scikit-learn in c:\users\mttvi\anaconda3\lib\site-packages (1.0.2)
    Collecting scikit-learn
      Downloading scikit_learn-1.5.2-cp39-cp39-win_amd64.whl (11.0 MB)
         ---------------------------------------- 11.0/11.0 MB 2.5 MB/s eta 0:00:00
    Collecting joblib>=1.2.0
      Downloading joblib-1.4.2-py3-none-any.whl (301 kB)
         -------------------------------------- 301.8/301.8 kB 2.7 MB/s eta 0:00:00
    Requirement already satisfied: scipy>=1.6.0 in c:\users\mttvi\anaconda3\lib\site-packages (from scikit-learn) (1.11.4)
    Collecting threadpoolctl>=3.1.0
      Downloading threadpoolctl-3.5.0-py3-none-any.whl (18 kB)
    Collecting numpy
      Downloading numpy-1.26.4-cp39-cp39-win_amd64.whl (15.8 MB)
         ---------------------------------------- 15.8/15.8 MB 2.2 MB/s eta 0:00:00
    Installing collected packages: threadpoolctl, numpy, joblib, scikit-learn
      Attempting uninstall: threadpoolctl
        Found existing installation: threadpoolctl 2.2.0
        Uninstalling threadpoolctl-2.2.0:
          Successfully uninstalled threadpoolctl-2.2.0
      Attempting uninstall: numpy
        Found existing installation: numpy 1.21.6
        Uninstalling numpy-1.21.6:
          Successfully uninstalled numpy-1.21.6
      Attempting uninstall: joblib
        Found existing installation: joblib 1.1.0
        Uninstalling joblib-1.1.0:
          Successfully uninstalled joblib-1.1.0
      Attempting uninstall: scikit-learn
        Found existing installation: scikit-learn 1.0.2
        Uninstalling scikit-learn-1.0.2:
          Successfully uninstalled scikit-learn-1.0.2
    Successfully installed joblib-1.4.2 numpy-1.26.4 scikit-learn-1.5.2 threadpoolctl-3.5.0
    Note: you may need to restart the kernel to use updated packages.
    

    WARNING: Ignoring invalid distribution -otobuf (c:\users\mttvi\anaconda3\lib\site-packages)
    WARNING: Ignoring invalid distribution -rotobuf (c:\users\mttvi\anaconda3\lib\site-packages)
    WARNING: Ignoring invalid distribution - (c:\users\mttvi\anaconda3\lib\site-packages)
    WARNING: Ignoring invalid distribution -otobuf (c:\users\mttvi\anaconda3\lib\site-packages)
    WARNING: Ignoring invalid distribution -rotobuf (c:\users\mttvi\anaconda3\lib\site-packages)
    WARNING: Ignoring invalid distribution - (c:\users\mttvi\anaconda3\lib\site-packages)
    WARNING: Ignoring invalid distribution -otobuf (c:\users\mttvi\anaconda3\lib\site-packages)
    WARNING: Ignoring invalid distribution -rotobuf (c:\users\mttvi\anaconda3\lib\site-packages)
    WARNING: Ignoring invalid distribution - (c:\users\mttvi\anaconda3\lib\site-packages)
        WARNING: Ignoring invalid distribution -otobuf (c:\users\mttvi\anaconda3\lib\site-packages)
        WARNING: Ignoring invalid distribution -rotobuf (c:\users\mttvi\anaconda3\lib\site-packages)
        WARNING: Ignoring invalid distribution - (c:\users\mttvi\anaconda3\lib\site-packages)
        WARNING: Ignoring invalid distribution -otobuf (c:\users\mttvi\anaconda3\lib\site-packages)
        WARNING: Ignoring invalid distribution -rotobuf (c:\users\mttvi\anaconda3\lib\site-packages)
        WARNING: Ignoring invalid distribution - (c:\users\mttvi\anaconda3\lib\site-packages)
        WARNING: Ignoring invalid distribution -otobuf (c:\users\mttvi\anaconda3\lib\site-packages)
        WARNING: Ignoring invalid distribution -rotobuf (c:\users\mttvi\anaconda3\lib\site-packages)
        WARNING: Ignoring invalid distribution - (c:\users\mttvi\anaconda3\lib\site-packages)
        WARNING: Ignoring invalid distribution -otobuf (c:\users\mttvi\anaconda3\lib\site-packages)
        WARNING: Ignoring invalid distribution -rotobuf (c:\users\mttvi\anaconda3\lib\site-packages)
        WARNING: Ignoring invalid distribution - (c:\users\mttvi\anaconda3\lib\site-packages)
    WARNING: Ignoring invalid distribution -otobuf (c:\users\mttvi\anaconda3\lib\site-packages)
    WARNING: Ignoring invalid distribution -rotobuf (c:\users\mttvi\anaconda3\lib\site-packages)
    WARNING: Ignoring invalid distribution - (c:\users\mttvi\anaconda3\lib\site-packages)
    WARNING: Ignoring invalid distribution -otobuf (c:\users\mttvi\anaconda3\lib\site-packages)
    WARNING: Ignoring invalid distribution -rotobuf (c:\users\mttvi\anaconda3\lib\site-packages)
    WARNING: Ignoring invalid distribution - (c:\users\mttvi\anaconda3\lib\site-packages)
    WARNING: Ignoring invalid distribution -otobuf (c:\users\mttvi\anaconda3\lib\site-packages)
    WARNING: Ignoring invalid distribution -rotobuf (c:\users\mttvi\anaconda3\lib\site-packages)
    WARNING: Ignoring invalid distribution - (c:\users\mttvi\anaconda3\lib\site-packages)
    WARNING: Ignoring invalid distribution -otobuf (c:\users\mttvi\anaconda3\lib\site-packages)
    WARNING: Ignoring invalid distribution -rotobuf (c:\users\mttvi\anaconda3\lib\site-packages)
    WARNING: Ignoring invalid distribution - (c:\users\mttvi\anaconda3\lib\site-packages)
    ERROR: pip's dependency resolver does not currently take into account all the packages that are installed. This behaviour is the source of the following dependency conflicts.
    daal4py 2021.6.0 requires daal==2021.4.0, which is not installed.
    streamlit 1.29.0 requires packaging<24,>=16.8, but you have packaging 24.1 which is incompatible.
    streamlit 1.29.0 requires protobuf<5,>=3.20, but you have protobuf 3.19.6 which is incompatible.
    numba 0.55.1 requires numpy<1.22,>=1.18, but you have numpy 1.26.4 which is incompatible.
    WARNING: Ignoring invalid distribution -otobuf (c:\users\mttvi\anaconda3\lib\site-packages)
    WARNING: Ignoring invalid distribution -rotobuf (c:\users\mttvi\anaconda3\lib\site-packages)
    WARNING: Ignoring invalid distribution - (c:\users\mttvi\anaconda3\lib\site-packages)
    WARNING: Ignoring invalid distribution -otobuf (c:\users\mttvi\anaconda3\lib\site-packages)
    WARNING: Ignoring invalid distribution -rotobuf (c:\users\mttvi\anaconda3\lib\site-packages)
    WARNING: Ignoring invalid distribution - (c:\users\mttvi\anaconda3\lib\site-packages)
    WARNING: Ignoring invalid distribution -otobuf (c:\users\mttvi\anaconda3\lib\site-packages)
    WARNING: Ignoring invalid distribution -rotobuf (c:\users\mttvi\anaconda3\lib\site-packages)
    WARNING: Ignoring invalid distribution - (c:\users\mttvi\anaconda3\lib\site-packages)
    


```python
pip install numpy==1.21
```

    Collecting numpy==1.21
      Downloading numpy-1.21.0-cp39-cp39-win_amd64.whl (14.0 MB)
         ---------------------------------------- 14.0/14.0 MB 3.3 MB/s eta 0:00:00
    Installing collected packages: numpy
      Attempting uninstall: numpy
        Found existing installation: numpy 1.26.4
        Uninstalling numpy-1.26.4:
          Successfully uninstalled numpy-1.26.4
    Successfully installed numpy-1.21.0
    Note: you may need to restart the kernel to use updated packages.
    

    WARNING: Ignoring invalid distribution -otobuf (c:\users\mttvi\anaconda3\lib\site-packages)
    WARNING: Ignoring invalid distribution -rotobuf (c:\users\mttvi\anaconda3\lib\site-packages)
    WARNING: Ignoring invalid distribution - (c:\users\mttvi\anaconda3\lib\site-packages)
    WARNING: Ignoring invalid distribution -otobuf (c:\users\mttvi\anaconda3\lib\site-packages)
    WARNING: Ignoring invalid distribution -rotobuf (c:\users\mttvi\anaconda3\lib\site-packages)
    WARNING: Ignoring invalid distribution - (c:\users\mttvi\anaconda3\lib\site-packages)
    WARNING: Ignoring invalid distribution -otobuf (c:\users\mttvi\anaconda3\lib\site-packages)
    WARNING: Ignoring invalid distribution -rotobuf (c:\users\mttvi\anaconda3\lib\site-packages)
    WARNING: Ignoring invalid distribution - (c:\users\mttvi\anaconda3\lib\site-packages)
        WARNING: Ignoring invalid distribution -otobuf (c:\users\mttvi\anaconda3\lib\site-packages)
        WARNING: Ignoring invalid distribution -rotobuf (c:\users\mttvi\anaconda3\lib\site-packages)
        WARNING: Ignoring invalid distribution - (c:\users\mttvi\anaconda3\lib\site-packages)
    WARNING: Ignoring invalid distribution -otobuf (c:\users\mttvi\anaconda3\lib\site-packages)
    WARNING: Ignoring invalid distribution -rotobuf (c:\users\mttvi\anaconda3\lib\site-packages)
    WARNING: Ignoring invalid distribution - (c:\users\mttvi\anaconda3\lib\site-packages)
    ERROR: pip's dependency resolver does not currently take into account all the packages that are installed. This behaviour is the source of the following dependency conflicts.
    daal4py 2021.6.0 requires daal==2021.4.0, which is not installed.
    streamlit 1.29.0 requires packaging<24,>=16.8, but you have packaging 24.1 which is incompatible.
    streamlit 1.29.0 requires protobuf<5,>=3.20, but you have protobuf 3.19.6 which is incompatible.
    scipy 1.11.4 requires numpy<1.28.0,>=1.21.6, but you have numpy 1.21.0 which is incompatible.
    pandas 2.2.3 requires numpy>=1.22.4; python_version < "3.11", but you have numpy 1.21.0 which is incompatible.
    nibabel 5.3.0 requires numpy>=1.22, but you have numpy 1.21.0 which is incompatible.
    faiss-cpu 1.9.0 requires numpy<3.0,>=1.25.0, but you have numpy 1.21.0 which is incompatible.
    WARNING: Ignoring invalid distribution -otobuf (c:\users\mttvi\anaconda3\lib\site-packages)
    WARNING: Ignoring invalid distribution -rotobuf (c:\users\mttvi\anaconda3\lib\site-packages)
    WARNING: Ignoring invalid distribution - (c:\users\mttvi\anaconda3\lib\site-packages)
    WARNING: Ignoring invalid distribution -otobuf (c:\users\mttvi\anaconda3\lib\site-packages)
    WARNING: Ignoring invalid distribution -rotobuf (c:\users\mttvi\anaconda3\lib\site-packages)
    WARNING: Ignoring invalid distribution - (c:\users\mttvi\anaconda3\lib\site-packages)
    WARNING: Ignoring invalid distribution -otobuf (c:\users\mttvi\anaconda3\lib\site-packages)
    WARNING: Ignoring invalid distribution -rotobuf (c:\users\mttvi\anaconda3\lib\site-packages)
    WARNING: Ignoring invalid distribution - (c:\users\mttvi\anaconda3\lib\site-packages)
    


```python
pip install --force-reinstall numpy scikit-learn
```

    Collecting numpy
      Using cached numpy-2.0.2-cp39-cp39-win_amd64.whl (15.9 MB)
    Collecting scikit-learn
      Using cached scikit_learn-1.5.2-cp39-cp39-win_amd64.whl (11.0 MB)
    Collecting scipy>=1.6.0
      Downloading scipy-1.13.1-cp39-cp39-win_amd64.whl (46.2 MB)
         ---------------------------------------- 46.2/46.2 MB 1.6 MB/s eta 0:00:00
    Collecting joblib>=1.2.0
      Using cached joblib-1.4.2-py3-none-any.whl (301 kB)
    Collecting threadpoolctl>=3.1.0
      Using cached threadpoolctl-3.5.0-py3-none-any.whl (18 kB)
    Installing collected packages: threadpoolctl, numpy, joblib, scipy, scikit-learn
      Attempting uninstall: threadpoolctl
        Found existing installation: threadpoolctl 3.5.0
        Uninstalling threadpoolctl-3.5.0:
          Successfully uninstalled threadpoolctl-3.5.0
      Attempting uninstall: numpy
        Found existing installation: numpy 1.21.0
        Uninstalling numpy-1.21.0:
          Successfully uninstalled numpy-1.21.0
      Attempting uninstall: joblib
        Found existing installation: joblib 1.4.2
        Uninstalling joblib-1.4.2:
          Successfully uninstalled joblib-1.4.2
      Attempting uninstall: scipy
        Found existing installation: scipy 1.11.4
        Uninstalling scipy-1.11.4:
          Successfully uninstalled scipy-1.11.4
      Attempting uninstall: scikit-learn
        Found existing installation: scikit-learn 1.5.2
        Uninstalling scikit-learn-1.5.2:
          Successfully uninstalled scikit-learn-1.5.2
    Successfully installed joblib-1.4.2 numpy-2.0.2 scikit-learn-1.5.2 scipy-1.13.1 threadpoolctl-3.5.0
    Note: you may need to restart the kernel to use updated packages.
    

    WARNING: Ignoring invalid distribution -otobuf (c:\users\mttvi\anaconda3\lib\site-packages)
    WARNING: Ignoring invalid distribution -rotobuf (c:\users\mttvi\anaconda3\lib\site-packages)
    WARNING: Ignoring invalid distribution - (c:\users\mttvi\anaconda3\lib\site-packages)
    WARNING: Ignoring invalid distribution -otobuf (c:\users\mttvi\anaconda3\lib\site-packages)
    WARNING: Ignoring invalid distribution -rotobuf (c:\users\mttvi\anaconda3\lib\site-packages)
    WARNING: Ignoring invalid distribution - (c:\users\mttvi\anaconda3\lib\site-packages)
    WARNING: Ignoring invalid distribution -otobuf (c:\users\mttvi\anaconda3\lib\site-packages)
    WARNING: Ignoring invalid distribution -rotobuf (c:\users\mttvi\anaconda3\lib\site-packages)
    WARNING: Ignoring invalid distribution - (c:\users\mttvi\anaconda3\lib\site-packages)
        WARNING: Ignoring invalid distribution -otobuf (c:\users\mttvi\anaconda3\lib\site-packages)
        WARNING: Ignoring invalid distribution -rotobuf (c:\users\mttvi\anaconda3\lib\site-packages)
        WARNING: Ignoring invalid distribution - (c:\users\mttvi\anaconda3\lib\site-packages)
        WARNING: Ignoring invalid distribution -otobuf (c:\users\mttvi\anaconda3\lib\site-packages)
        WARNING: Ignoring invalid distribution -rotobuf (c:\users\mttvi\anaconda3\lib\site-packages)
        WARNING: Ignoring invalid distribution - (c:\users\mttvi\anaconda3\lib\site-packages)
        WARNING: Ignoring invalid distribution -otobuf (c:\users\mttvi\anaconda3\lib\site-packages)
        WARNING: Ignoring invalid distribution -rotobuf (c:\users\mttvi\anaconda3\lib\site-packages)
        WARNING: Ignoring invalid distribution - (c:\users\mttvi\anaconda3\lib\site-packages)
        WARNING: Ignoring invalid distribution -otobuf (c:\users\mttvi\anaconda3\lib\site-packages)
        WARNING: Ignoring invalid distribution -rotobuf (c:\users\mttvi\anaconda3\lib\site-packages)
        WARNING: Ignoring invalid distribution - (c:\users\mttvi\anaconda3\lib\site-packages)
        WARNING: Ignoring invalid distribution -otobuf (c:\users\mttvi\anaconda3\lib\site-packages)
        WARNING: Ignoring invalid distribution -rotobuf (c:\users\mttvi\anaconda3\lib\site-packages)
        WARNING: Ignoring invalid distribution - (c:\users\mttvi\anaconda3\lib\site-packages)
    WARNING: Ignoring invalid distribution -otobuf (c:\users\mttvi\anaconda3\lib\site-packages)
    WARNING: Ignoring invalid distribution -rotobuf (c:\users\mttvi\anaconda3\lib\site-packages)
    WARNING: Ignoring invalid distribution - (c:\users\mttvi\anaconda3\lib\site-packages)
    WARNING: Ignoring invalid distribution -otobuf (c:\users\mttvi\anaconda3\lib\site-packages)
    WARNING: Ignoring invalid distribution -rotobuf (c:\users\mttvi\anaconda3\lib\site-packages)
    WARNING: Ignoring invalid distribution - (c:\users\mttvi\anaconda3\lib\site-packages)
    WARNING: Ignoring invalid distribution -otobuf (c:\users\mttvi\anaconda3\lib\site-packages)
    WARNING: Ignoring invalid distribution -rotobuf (c:\users\mttvi\anaconda3\lib\site-packages)
    WARNING: Ignoring invalid distribution - (c:\users\mttvi\anaconda3\lib\site-packages)
    WARNING: Ignoring invalid distribution -otobuf (c:\users\mttvi\anaconda3\lib\site-packages)
    WARNING: Ignoring invalid distribution -rotobuf (c:\users\mttvi\anaconda3\lib\site-packages)
    WARNING: Ignoring invalid distribution - (c:\users\mttvi\anaconda3\lib\site-packages)
    WARNING: Ignoring invalid distribution -otobuf (c:\users\mttvi\anaconda3\lib\site-packages)
    WARNING: Ignoring invalid distribution -rotobuf (c:\users\mttvi\anaconda3\lib\site-packages)
    WARNING: Ignoring invalid distribution - (c:\users\mttvi\anaconda3\lib\site-packages)
    ERROR: pip's dependency resolver does not currently take into account all the packages that are installed. This behaviour is the source of the following dependency conflicts.
    daal4py 2021.6.0 requires daal==2021.4.0, which is not installed.
    streamlit 1.29.0 requires numpy<2,>=1.19.3, but you have numpy 2.0.2 which is incompatible.
    streamlit 1.29.0 requires packaging<24,>=16.8, but you have packaging 24.1 which is incompatible.
    streamlit 1.29.0 requires protobuf<5,>=3.20, but you have protobuf 3.19.6 which is incompatible.
    numba 0.55.1 requires numpy<1.22,>=1.18, but you have numpy 2.0.2 which is incompatible.
    langchain 0.3.3 requires numpy<2,>=1; python_version < "3.12", but you have numpy 2.0.2 which is incompatible.
    WARNING: Ignoring invalid distribution -otobuf (c:\users\mttvi\anaconda3\lib\site-packages)
    WARNING: Ignoring invalid distribution -rotobuf (c:\users\mttvi\anaconda3\lib\site-packages)
    WARNING: Ignoring invalid distribution - (c:\users\mttvi\anaconda3\lib\site-packages)
    WARNING: Ignoring invalid distribution -otobuf (c:\users\mttvi\anaconda3\lib\site-packages)
    WARNING: Ignoring invalid distribution -rotobuf (c:\users\mttvi\anaconda3\lib\site-packages)
    WARNING: Ignoring invalid distribution - (c:\users\mttvi\anaconda3\lib\site-packages)
    WARNING: Ignoring invalid distribution -otobuf (c:\users\mttvi\anaconda3\lib\site-packages)
    WARNING: Ignoring invalid distribution -rotobuf (c:\users\mttvi\anaconda3\lib\site-packages)
    WARNING: Ignoring invalid distribution - (c:\users\mttvi\anaconda3\lib\site-packages)
    


```python
import numpy as np

# Define a function to compute cosine similarity
def cosine_similarity(embedding1, embedding2):
    # Normalize vectors
    norm1 = np.linalg.norm(embedding1)
    norm2 = np.linalg.norm(embedding2)
    
    # Handle division by zero
    if norm1 == 0 or norm2 == 0:
        return 0
    
    # Compute the dot product and divide by norms
    similarity = np.dot(embedding1, embedding2) / (norm1 * norm2)
    return similarity

# Example usage with two embeddings
embedding1 = np.array([1, 2, 3])
embedding2 = np.array([4, 5, 6])
similarity = cosine_similarity(embedding1, embedding2)

print(f"Cosine Similarity: {similarity}")

```

    Cosine Similarity: 0.9746318461970762
    


```python

```
