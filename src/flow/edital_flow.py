from typing import Callable
from crewai import Flow, LLM
import os
from crewai.flow.flow import start, listen
from utils import read_metadata
from tools.file_tools import FileReadTool, DocumentTooLargeError, InsufficientContentError
from flow.models import TargetAnalysis, QuantitiesAnalysis, SummaryAnalysis, EditalState
import json
import functools
from utils import get_logger

logger = get_logger(__name__)


def handle_flow_error(func: Callable) -> Callable:
    """Decorator para tratar erros no flow de forma consistente."""
    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        print(f"\n{'='*50}\nExecutando {func.__name__}")
        print(f"Estado atual - has_error: {self.state.has_error}, error_message: {self.state.error_message}")
        
        # Se já houve erro, apenas retorna
        if self.state.has_error:
            print(f"IGNORADO: {func.__name__} - Erro anterior: {self.state.error_message}")
            return

        try:
            result = func(self, *args, **kwargs)
            print(f"✅ {func.__name__} executado com sucesso")
            return result
        except DocumentTooLargeError as e:
            print(f"❌ {func.__name__} - DocumentTooLargeError: {str(e)}")
            self.state.has_error = True
            self.state.error_message = str(e)
            self.state.justification = e.error_message
            self.state.target_match = False
            self.state.threshold_match = "inconclusive"
            self.state.is_relevant = False
            return self.state
        except InsufficientContentError as e:
            print(f"❌ {func.__name__} - InsufficientContentError: {str(e)}")
            self.state.has_error = True
            self.state.error_message = str(e)
            self.state.justification = f"Não foi possível analisar o edital: {e.error_message}. É necessário ter arquivos de conteúdo (PDF, DOCX, etc.) além do metadata.json para realizar a análise."
            self.state.target_match = False
            self.state.threshold_match = "inconclusive"
            self.state.is_relevant = False
            return self.state
        except Exception as e:
            print(f"❌ {func.__name__} - Exception: {str(e)}")
            self.state.has_error = True
            self.state.error_message = f"Erro em {func.__name__}: {str(e)}"
            self.state.justification = self.state.error_message
            return self.state
    return wrapper



def create_llm(
    *,
    temperature: float,
    max_tokens: int,
    top_p: float,
    frequency_penalty: float,
    presence_penalty: float,
    response_format=None,
):
    """Cria uma instância de LLM configurável via variáveis de ambiente.

    Variáveis suportadas:
    - LLM_MODEL: nome do modelo (ex: openai/gpt-4o, ollama/gpt-oss:20b)
    - LLM_BASE_URL: ex: http://localhost:11434/v1 (para Ollama - rota OpenAI compatível)
    """
    model = os.getenv("LLM_MODEL", "").strip()
    base_url = os.getenv("LLM_BASE_URL", "").strip()
    
    print(f"🔧 create_llm: Configuração do LLM:")
    print(f"🔧 create_llm: - Model: '{model}'")
    print(f"🔧 create_llm: - Base URL: '{base_url}'")
    print(f"🔧 create_llm: - Temperature: {temperature}")
    print(f"🔧 create_llm: - Max tokens: {max_tokens}")
    print(f"🔧 create_llm: - Response format: {response_format}")
    
    if not model:
        print(f"❌ create_llm: LLM_MODEL não definido!")
        raise ValueError("LLM_MODEL não definido nas variáveis de ambiente")
    
    if not base_url and "ollama" in model.lower():
        print(f"⚠️ create_llm: LLM_BASE_URL não definido para modelo Ollama, usando padrão")
        base_url = "http://localhost:11434/v1"

    llm = LLM(
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=top_p,
        frequency_penalty=frequency_penalty,
        presence_penalty=presence_penalty,
        # response_format=response_format,
        base_url=(base_url or None),
        timeout=1800 # 30 minutos
    )
    print(f"✅ create_llm: LLM criado com sucesso")
    return llm


class EditalAnalysisFlow(Flow[EditalState]):
    """Flow para análise de editais."""

    def __init__(self, target: str, threshold: int = 0, force_match: bool = False):
        super().__init__()
        self.target = target
        self.threshold = threshold
        self.force_match = force_match
        self.file_tool = FileReadTool()
        logger.info(f"Target: {target}")
        logger.info(f"Threshold: {threshold}")
        logger.info(f"Forçar Match: {force_match}")

    @start()
    @handle_flow_error
    def extract_metadata(self):
        """Extrai metadata do edital."""
        metadata = read_metadata(self.state.edital_path_dir)
        self.state.metadata = metadata
        
        # Extrai bid_number do metadata
        if "bid_number" in metadata:
            self.state.bid_number = metadata["bid_number"]
        

    @listen(extract_metadata)
    @handle_flow_error
    def extract_content(self):
        """Extrai conteúdo do edital."""
        try:
            # Usa o FileReadTool para ler todos os arquivos do diretório
            content = self.file_tool._run(self.state.edital_path_dir)
            # NOVO: Se o conteúdo for vazio ou só erro, aborta o fluxo
            if not content or content.strip().startswith("Error") or not content.strip():
                self.state.has_error = True
                self.state.error_message = "Nenhum conteúdo extraído dos arquivos do edital."
                self.state.justification = (
                    "Não foi possível analisar o edital devido à ausência de conteúdo nos arquivos. "
                    "Verifique se os arquivos estão corrompidos, vazios ou ilegíveis."
                )
                self.state.target_match = False
                self.state.threshold_match = "inconclusive"
                self.state.is_relevant = False
                self.state.executive_summary = "Não foi possível gerar resumo: nenhum conteúdo extraído."
                self.state.technical_summary = "Não foi possível gerar resumo técnico: nenhum conteúdo extraído."
                return self.state
            self.state.content = content
        except DocumentTooLargeError as e:
            self.state.has_error = True
            self.state.error_message = str(e)
            self.state.justification = e.error_message
            self.state.target_match = False
            self.state.threshold_match = "inconclusive"
            self.state.is_relevant = False
            return self.state
        except InsufficientContentError as e:
            self.state.has_error = True
            self.state.error_message = str(e)
            self.state.justification = f"Não foi possível analisar o edital: {e.error_message}. É necessário ter arquivos de conteúdo (PDF, DOCX, etc.) além do metadata.json para realizar a análise."
            self.state.target_match = False
            self.state.threshold_match = "inconclusive"
            self.state.is_relevant = False
            return self.state
        except FileNotFoundError as e:
            self.state.has_error = True
            self.state.error_message = str(e)
            self.state.justification = f"Erro ao processar edital: {str(e)}"
            return self.state
        except Exception as e:
            print(f"❌ Erro ao extrair conteúdo: {str(e)}")
            self.state.has_error = True
            self.state.error_message = f"Erro ao extrair conteúdo: {str(e)}"
            self.state.justification = self.state.error_message
            return self.state

    @listen(extract_content)
    @handle_flow_error
    def generate_summary(self):
        """Gera resumo executivo e técnico do edital."""
        llm = create_llm(
            temperature=0.7,
            max_tokens=2500,
            top_p=0.9,
            frequency_penalty=0.1,
            presence_penalty=0.1,
            response_format=SummaryAnalysis,
        )

        # Faz a chamada ao LLM
        print(f"🔍 generate_summary: Iniciando chamada ao LLM...")
        print(f"🔍 generate_summary: Modelo: {os.getenv('LLM_MODEL', 'N/A')}")
        print(f"🔍 generate_summary: Base URL: {os.getenv('LLM_BASE_URL', 'N/A')}")
        print(f"🔍 generate_summary: Tamanho do conteúdo: {len(self.state.content)} caracteres")
        print(f"🔍 generate_summary: Tamanho do metadata: {len(json.dumps(self.state.metadata, ensure_ascii=False))} caracteres")
        
        try:
            response = llm.call(
                messages=[
                    {"role": "system", "content": """Você é um especialista em resumir editais de licitação.
                    Sua tarefa é gerar DOIS tipos de resumo: executivo e técnico, ambos em formato Markdown e em PORTUGUÊS.
                    
                    RESUMO EXECUTIVO (executive_summary):
                    - Foco comercial e direto
                    - Máximo 150 palavras
                    - Formato Markdown limpo e profissional
                    - NÃO adicione nenhum título como 'Resumo Executivo' ou similar
                    - Incluir:
                      * Cidade/UF
                      * Valor Monetário total estimado (se disponível)
                      * Data de abertura (com data e hora)
                      * Quantidades principais (resumidas)
                      * Contatos essenciais
                    - NÃO incluir especificações técnicas ou detalhes operacionais
                    - Linguagem comercial e acessível
                    
                    RESUMO TÉCNICO (technical_summary):
                    - Detalhado e completo
                    - Máximo 1500 palavras
                    - Formato Markdown estruturado e profissional
                    - NÃO adicione nenhum título como 'Resumo Técnico' ou similar
                    - Incluir:
                      * Informações gerais completas
                      * Especificações técnicas detalhadas
                      * Quantidades e valores
                      * Prazos e condições
                      * Requisitos técnicos
                      * Condições comerciais
                      * Contatos completos
                    - Usar títulos em português para as seções internas, mas nunca um título geral como 'Resumo Técnico'
                    - Manter consistência na formatação
                    
                    REGRAS GERAIS:
                    - SEMPRE em português
                    - Formatação Markdown consistente
                    - Títulos com # e ## apenas para seções internas, nunca para o resumo como um todo
                    - Listas com - e **negrito** para destaque
                    - Informações organizadas em seções claras
                    - Linguagem profissional mas acessível
                    - Nunca adicione um título geral como 'Resumo Técnico' ou 'Resumo Executivo' no início do texto
                    
                    IMPORTANTE: Identifique também:
                    1. A cidade/UF do edital (procure no metadata ou no texto)
                    2. A data de abertura (opening_date) no formato DD/MM/YYYY (apenas a data, sem hora)
                    3. Informações de contato:
                       - phone: telefone de contato
                       - website: site oficial
                       - email: email de contato
                    4. Outras informações (em formato texto):
                       - title: título do edital
                       - object: objeto da licitação
                       - quantities: quantidades relevantes (ex: '100 unidades de tablets')
                       - specifications: especificações técnicas (ex: 'Tablets com tela de 10 polegadas')
                       - deadlines: prazos importantes (ex: 'Entrega em 30 dias')
                       - values: valores relevantes (ex: 'Valor total de R$ 500.000,00')
                    Se não encontrar alguma informação, use string vazia"""},
                    {"role": "user", "content": f"""
                    Target: {self.target}
                    
                    Metadata:
                    {json.dumps(self.state.metadata, ensure_ascii=False, indent=2)}
                    
                    Conteúdo do Edital:
                    {self.state.content}
                    """}
                ]
            )
            
            print(f"🔍 generate_summary: Resposta recebida do LLM")
            print(f"🔍 generate_summary: Tamanho da resposta: {len(response) if response else 0} caracteres")
            print(f"🔍 generate_summary: Primeiros 200 caracteres da resposta: {response[:200] if response else 'RESPOSTA VAZIA'}")
            
            if not response or not response.strip():
                print(f"❌ generate_summary: Resposta vazia do LLM")
                raise ValueError("Resposta vazia do LLM")
            
            # Tenta fazer parse do JSON
            print(f"🔍 generate_summary: Tentando fazer parse do JSON...")
            response_dict = json.loads(response)
            print(f"✅ generate_summary: JSON parseado com sucesso")
            
        except json.JSONDecodeError as e:
            print(f"❌ generate_summary: Erro ao fazer parse do JSON: {str(e)}")
            print(f"❌ generate_summary: Resposta completa: {response}")
            raise ValueError(f"Resposta do LLM não é um JSON válido: {str(e)}")
        except Exception as e:
            print(f"❌ generate_summary: Erro na chamada ao LLM: {str(e)}")
            raise
        response_obj = SummaryAnalysis(**response_dict)
        cleaned_data = response_obj.clean_empty_fields()
        
        # Atualiza o state
        self.state.executive_summary = cleaned_data.get("executive_summary", "")
        self.state.technical_summary = cleaned_data.get("technical_summary", "")
        self.state.city = cleaned_data.get("city", "Não foi possível determinar")
        self.state.opening_date = cleaned_data.get("opening_date", "")
        
        # Corrige o campo title se vier igual ao target
        title = cleaned_data.get("title", "")
        if title.strip().lower() == self.target.strip().lower():
            title = "Não disponível"
        
        # Atualiza o metadata com as informações de contato e outras informações
        self.state.metadata.update({
            "phone": cleaned_data.get("phone", ""),
            "website": cleaned_data.get("website", ""),
            "email": cleaned_data.get("email", ""),
            "title": title,
            "object": cleaned_data.get("object", ""),
            "quantities": cleaned_data.get("quantities", ""),
            "specifications": cleaned_data.get("specifications", ""),
            "deadlines": cleaned_data.get("deadlines", ""),
            "values": cleaned_data.get("values", "")
        })
        

    @listen(generate_summary)
    @handle_flow_error
    def analyze_target(self):
        """Analisa se o edital é relevante para o target."""
        llm = create_llm(
            temperature=0.7,
            max_tokens=1000,
            top_p=0.9,
            frequency_penalty=0.1,
            presence_penalty=0.1,
            response_format=TargetAnalysis,
        )

        response = llm.call(
            messages=[
                {"role": "system", "content": """Você é um especialista em análise de editais de licitação.
                Sua tarefa é analisar se o edital é relevante para o target especificado.
                
                Considere:
                1. Sinônimos e termos relacionados
                2. Especificações técnicas que indiquem o tipo de equipamento
                3. Contexto do uso do equipamento/serviço"""},
                {"role": "user", "content": f"""
                Target: {self.target}
                
                Resumo Técnico do Edital:
                {self.state.technical_summary}
                
                Informações Adicionais:
                - Objeto: {self.state.metadata.get('object', '')}
                - Especificações: {self.state.metadata.get('specifications', '')}
                - Quantidades: {self.state.metadata.get('quantities', '')}
                """}
            ]
        )
        
        response_dict = json.loads(response)
        response_obj = TargetAnalysis(**response_dict)
        
        self.state.target_match = response_obj.is_relevant
        
        if self.force_match:
            self.state.target_match = True
        

    @listen(analyze_target)
    @handle_flow_error
    def check_threshold(self):
        """Verifica se o edital atende ao threshold."""
        
        if self.threshold == 0:
            self.state.threshold_match = "true"
            return

        if not self.state.target_match:
            self.state.threshold_match = "false"
            return

        llm = create_llm(
            temperature=0.7,
            max_tokens=1000,
            top_p=0.9,
            frequency_penalty=0.1,
            presence_penalty=0.1,
            response_format=QuantitiesAnalysis,
        )

        response = llm.call(
            messages=[
                {"role": "system", "content": """Você é um especialista em análise de quantidades em editais.
                Sua tarefa é identificar e somar todas as quantidades relevantes para o target.
                
                Instruções:
                1. Analise cuidadosamente o resumo do edital
                2. Identifique TODAS as quantidades mencionadas
                3. Some as quantidades relevantes para o target
                4. Especifique a unidade de medida
                5. Explique como chegou na quantidade total
                
                Exemplo de resposta:
                {
                    "total_quantity": 100,
                    "unit": "unidades",
                    "explanation": "Encontrei 100 unidades mencionadas no edital"
                }"""},
                {"role": "user", "content": f"""
                Target: {self.target}
                Threshold: {self.threshold}
                
                Resumo Técnico do Edital:
                {self.state.technical_summary}
                """}
            ]
        )
        
        # Validação 1: Verifica se a resposta é um JSON válido
        try:
            response_dict = json.loads(response)
        except json.JSONDecodeError as e:
            self.state.threshold_match = "inconclusive"
            self.state.metadata["quantities"] = "Erro ao processar quantidades: resposta inválida"
            return
        
        required_fields = ["total_quantity", "unit", "explanation"]
        missing_fields = [field for field in required_fields if field not in response_dict]
        if missing_fields:
            self.state.threshold_match = "inconclusive"
            self.state.metadata["quantities"] = f"Erro ao processar quantidades: campos ausentes ({', '.join(missing_fields)})"
            return
        
        try:
            total_quantity = int(response_dict["total_quantity"])
            if total_quantity < 0:
                raise ValueError("Quantidade não pode ser negativa")
        except (ValueError, TypeError) as e:
            self.state.threshold_match = "inconclusive"
            self.state.metadata["quantities"] = "Erro ao processar quantidades: valor inválido"
            return
        
        if not response_dict["unit"] or not isinstance(response_dict["unit"], str):
            self.state.threshold_match = "inconclusive"
            self.state.metadata["quantities"] = "Erro ao processar quantidades: unidade inválida"
            return
        
        response_obj = QuantitiesAnalysis(
            total_quantity=total_quantity,
            unit=response_dict["unit"],
            explanation=response_dict["explanation"]
        )
        
        if response_obj.total_quantity >= self.threshold:
            self.state.threshold_match = "true"
        else:
            self.state.threshold_match = "false"
        
        self.state.metadata["quantities"] = f"{response_obj.total_quantity} {response_obj.unit} - {response_obj.explanation}"

    @listen(check_threshold)
    @handle_flow_error
    def generate_justification(self):
        """Gera justificativa para a decisão."""
        llm = create_llm(
            temperature=0.7,
            max_tokens=200,
            top_p=0.9,
            frequency_penalty=0.1,
            presence_penalty=0.1,
        )

        response = llm.call(
            messages=[
                {"role": "system", "content": """Você é um especialista em justificar decisões sobre editais.
                Sua tarefa é gerar uma justificativa concisa, direta e profissional.
                
                Lógica de Decisão:
                1. Target Match (true/false):
                   - true: O edital menciona o target ou equipamentos/serviços similares
                   - false: O edital não menciona nada relacionado ao target
                
                2. Threshold Match ("true"/"false"/"inconclusive"):
                   - "true": Quantidade >= threshold
                   - "false": Quantidade < threshold
                   - "inconclusive": Não foi possível determinar a quantidade
                
                3. Edital Relevante (is_relevant):
                   - true: target_match=true E (threshold=0 OU threshold_match="true")
                   - false: target_match=false OU threshold_match="false"
                
                Instruções:
                1. Seja direto e objetivo
                2. Use linguagem profissional
                3. Máximo 100 palavras
                4. Se não relevante:
                   - Se target_match=false: Explique por que não encontrou o target
                   - Se threshold_match="false": Explique por que a quantidade é insuficiente
                5. Se relevante:
                   - Se threshold=0: Explique por que o target é relevante
                   - Se threshold>0: Explique por que a quantidade é suficiente
                6. Use português claro e acessível"""},
                {"role": "user", "content": f"""
                Target: {self.target}
                Target Match: {self.state.target_match}
                Threshold Match: {self.state.threshold_match}
                Threshold: {self.threshold}
                
                Resumo Executivo do Edital:
                {self.state.executive_summary}
                """}
            ]
        )
        
        self.state.justification = response
        
        self.state.is_relevant = (
            self.state.target_match and 
            (self.threshold == 0 or self.state.threshold_match == "true")
        )
        
def kickoff(edital_path_dir: str, target: str, threshold: int = 0, force_match: bool = False) -> EditalState:
    """Executa o fluxo de análise de edital."""
    
    flow = EditalAnalysisFlow(target=target, threshold=threshold, force_match=force_match)
    flow.kickoff({"edital_path_dir": edital_path_dir})
    logger.info("=== Flow Concluído ===")
    return flow.state

def plot():
    """Gera uma visualização do fluxo."""
    flow = EditalAnalysisFlow(target="exemplo", threshold=0)
    flow.plot("edital_analysis_flow")
    logger.info("Visualização do fluxo salva em edital_analysis_flow.html") 