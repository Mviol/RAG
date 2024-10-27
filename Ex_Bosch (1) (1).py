#!/usr/bin/env python
# coding: utf-8

# #### Resumo
# 
# O código começa importando as bibliotecas necessárias (fpdf e random).
# Define uma função gerar_dados() que cria 10.000 linhas de dados aleatórios sobre atendimentos, vendas, reclamações e elogios para diferentes meses.
# Cria um objeto PDF usando FPDF.
# Adiciona 5 páginas de texto introdutório sobre a Bosch.
# Gera uma grande tabela com os 10.000 registros de dados fictícios.
# Salva o PDF como 'relatorio_bosch.pdf'.
# Extração de texto e imagens do PDF:
# 
# Usa a biblioteca PyMuPDF (fitz) para abrir e processar o PDF.
# Define uma função extract_text_and_images_from_pdf() que extrai todo o texto e imagens do PDF.
# Imprime uma amostra do texto extraído e o número de imagens encontradas.
# Divisão do texto em chunks:
# 
# Importa o RecursiveCharacterTextSplitter da biblioteca langchain.
# Define um texto de exemplo.
# Configura o splitter para criar chunks de 500 caracteres com 50 caracteres de sobreposição.
# Divide o texto em chunks e os imprime.
# Configuração de modelos de linguagem e embeddings:
# 
# Importa um tokenizer e modelo GPT-J da biblioteca transformers (comentado no código).
# Configura um sistema de embeddings usando a biblioteca txtai.
# Cálculo de similaridade do cosseno:
# 
# Importa numpy para cálculos vetoriais.
# Define uma função cosine_similarity() que calcula a similaridade do cosseno entre dois vetores de embedding.
# Demonstra o uso da função com dois vetores de exemplo.

# # TESTE

# In[1]:


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


# In[2]:


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


# In[3]:


def create_chunks(text, chunk_size=100):
    words = text.split()
    return


# #### Funcionalidade para Extrair os Dados do Documento:

# In[4]:


import fitz  # PyMuPDF

def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page_num in range(doc.page_count):
        page = doc.load_page(page_num)
        text += page.get_text()
    return text


# ####  LLM Open Source e/ou na Nuvem:

# In[ ]:


from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B")
model = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-j-6B")


# ####  Base de Dados para os Embeddings:

# In[2]:


pip install txtai


# In[3]:


from txtai.embeddings import Embeddings

embeddings = Embeddings()
embeddings.index([("1", "Text data to index")])


# ####  Funcionalidade para Definir os "Chunks":

# In[19]:


pip install langchain


# In[24]:


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


# ####  Métrica de Avaliação da Qualidade:

# In[22]:


get_ipython().system('pip uninstall numpy scikit-learn')


# In[25]:


get_ipython().system('pip install numpy scikit-learn')


# In[26]:


get_ipython().system('pip install numpy==1.21.6 scikit-learn==1.0.2')


# In[27]:


get_ipython().system('pip cache purge')


# In[29]:


pip install --upgrade numpy scikit-learn


# In[30]:


pip install numpy==1.21


# In[31]:


pip install --force-reinstall numpy scikit-learn


# In[34]:


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


# In[ ]:




