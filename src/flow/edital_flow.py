from typing import Callable
from crewai import Flow
import os
import litellm
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



def get_ollama_config(temperature: float = 0.7, max_tokens: int = 2500):
    """Retorna configuração simplificada para Ollama."""
    model = os.getenv("LLM_MODEL", "llama3.2:3b").strip()
    base_url = os.getenv("LLM_BASE_URL", "http://localhost:11434").strip()
    
    print(f"🔧 get_ollama_config: Variáveis de ambiente:")
    print(f"🔧 get_ollama_config: - LLM_MODEL: '{os.getenv('LLM_MODEL', 'NÃO DEFINIDO')}'")
    print(f"🔧 get_ollama_config: - LLM_BASE_URL: '{os.getenv('LLM_BASE_URL', 'NÃO DEFINIDO')}'")
    print(f"🔧 get_ollama_config: - Model processado: '{model}'")
    print(f"🔧 get_ollama_config: - Base URL processada: '{base_url}'")
    
    # Remove 'ollama/' se já estiver presente no modelo
    if model.startswith("ollama/"):
        model = model[7:]  # Remove 'ollama/'
        print(f"🔧 get_ollama_config: - Model após remoção de 'ollama/': '{model}'")
    
    # URL do Ollama não usa /v1
    api_base = base_url
    model_name = f"ollama/{model}"
    
    print(f"🔧 get_ollama_config: URLs finais:")
    print(f"🔧 get_ollama_config: - API Base: '{api_base}'")
    print(f"🔧 get_ollama_config: - Model Name: '{model_name}'")
    
    config = {
        "model": model_name,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "api_base": api_base,
        "timeout": 1800,
    }
    
    print(f"🔧 get_ollama_config: Configuração criada: {config}")
    return config

def call_ollama(system_content: str, user_content: str, temperature: float = 0, max_tokens: int = 2500, response_format=None):
    """Faz chamada ao Ollama e retorna resposta limpa."""
    llm_config = get_ollama_config(temperature, max_tokens)
    
    # Se temos um modelo Pydantic, adiciona o schema JSON no formato LiteLLM
    if response_format:
        try:
            json_schema = response_format.model_json_schema()
            # llm_config["format"] = "json"
            llm_config["response_format"] = {
               "type": "json_schema",
               "json_schema": {
                   "schema": json_schema
               }
            }
            print(f"🔍 call_ollama: Schema JSON adicionado:")
            print(f"🔍 call_ollama: - Schema: {json_schema}")
        except Exception as e:
            print(f"⚠️ call_ollama: Erro ao gerar schema JSON: {str(e)}")
    
    print(f"🔍 call_ollama: Configuração final:")
    print(f"🔍 call_ollama: - Model: {llm_config['model']}")
    print(f"🔍 call_ollama: - API Base: {llm_config['api_base']}")
    print(f"🔍 call_ollama: - Temperature: {llm_config['temperature']}")
    print(f"🔍 call_ollama: - Max tokens: {llm_config['max_tokens']}")
    print(f"🔍 call_ollama: - Timeout: {llm_config['timeout']}")
    print(f"🔍 call_ollama: - Response Format: {llm_config.get('response_format', 'None')}")
    
    print(f"🔍 call_ollama: Fazendo chamada ao LiteLLM...")
    
    try:
        response = litellm.completion(
            messages=[
                {"role": "system", "content": system_content},
                {"role": "user", "content": user_content}
            ],
            **llm_config
        )
        
        print(f"✅ call_ollama: Resposta recebida com sucesso")
        print(f"🔍 call_ollama: Tipo da resposta: {type(response)}")
        print(f"🔍 call_ollama: Número de choices: {len(response.choices) if hasattr(response, 'choices') else 'N/A'}")
        
        return response.choices[0].message.content
        
    except Exception as e:
        print(f"❌ call_ollama: Erro na chamada:")
        print(f"❌ call_ollama: - Tipo do erro: {type(e).__name__}")
        print(f"❌ call_ollama: - Mensagem: {str(e)}")
        print(f"❌ call_ollama: - Configuração usada: {llm_config}")
        raise

def extract_json_from_text(text: str):
    """Tenta extrair JSON de um texto que pode conter outros conteúdos."""
    import re
    
    # Procura por JSON entre chaves
    json_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
    matches = re.findall(json_pattern, text, re.DOTALL)
    
    for match in matches:
        try:
            # Tenta fazer parse do JSON encontrado
            parsed = json.loads(match)
            print(f"🔍 extract_json_from_text: JSON encontrado e parseado: {list(parsed.keys())}")
            return parsed
        except json.JSONDecodeError:
            continue
    
    # Se não encontrou JSON válido, tenta encontrar o primeiro { até o último }
    start = text.find('{')
    end = text.rfind('}')
    
    if start != -1 and end != -1 and end > start:
        json_candidate = text[start:end+1]
        try:
            parsed = json.loads(json_candidate)
            print(f"🔍 extract_json_from_text: JSON extraído por posição: {list(parsed.keys())}")
            return parsed
        except json.JSONDecodeError:
            pass
    
    return None

def clean_and_parse_json(response: str, model_class):
    """Limpa resposta e faz parse para modelo Pydantic."""
    print(f"🔍 clean_and_parse_json: Resposta original:")
    print(f"🔍 clean_and_parse_json: - Tamanho: {len(response) if response else 0} caracteres")
    print(f"🔍 clean_and_parse_json: - Primeiros 200 chars: '{response[:200] if response else 'VAZIA'}'")
    
    # Com schema JSON estruturado, a resposta deve ser JSON puro
    cleaned = response.strip()
    print(f"🔍 clean_and_parse_json: Resposta limpa: '{cleaned[:200]}...'")
    
    if not cleaned:
        raise ValueError("Resposta vazia")
    
    # Tenta parse direto usando model_validate_json do Pydantic
    try:
        result = model_class.model_validate_json(cleaned)
        print(f"✅ clean_and_parse_json: JSON parseado com sucesso usando Pydantic: {list(result.model_dump().keys())}")
        return result
    except Exception as e:
        print(f"❌ clean_and_parse_json: Erro no parse JSON: {str(e)}")
        print(f"❌ clean_and_parse_json: Conteúdo: '{cleaned}'")
        
        # Fallback: tenta extrair JSON se ainda houver problema
        print(f"🔍 clean_and_parse_json: Tentando extrair JSON como fallback...")
        extracted_json = extract_json_from_text(cleaned)
        if extracted_json:
            try:
                result = model_class(**extracted_json)
                print(f"✅ clean_and_parse_json: JSON extraído e parseado com sucesso: {list(result.model_dump().keys())}")
                return result
            except Exception as e2:
                print(f"❌ clean_and_parse_json: Erro ao criar objeto Pydantic: {str(e2)}")
                raise ValueError(f"Resposta não contém JSON válido: {str(e)}")
        else:
            raise ValueError(f"Resposta não contém JSON válido: {str(e)}")


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
        print(f"🔍 generate_summary: Chamando Ollama...")
        
        system_content = """Você é um especialista em resumir editais de licitação.
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
        Se não encontrar alguma informação, use string vazia"""
        
        user_content = f"""
        Target: {self.target}
        
        Metadata:
        {json.dumps(self.state.metadata, ensure_ascii=False, indent=2)}
        
        Conteúdo do Edital:
        {self.state.content}
        """
        
        try:
            response = call_ollama(system_content, user_content, temperature=0, max_tokens=2500, response_format=SummaryAnalysis)
            print(f"✅ generate_summary: Resposta recebida do Ollama")
            
            if not response or not response.strip():
                raise ValueError("Resposta vazia do LLM")
            
            response_obj = clean_and_parse_json(response, SummaryAnalysis)
            cleaned_data = response_obj.clean_empty_fields()
            
        except Exception as e:
            print(f"❌ generate_summary: Erro ao processar resposta: {str(e)}")
            raise ValueError(f"Erro ao processar resposta do LLM: {str(e)}")
        
        # Atualiza o state
        self.state.executive_summary = cleaned_data.get("executive_summary", "")
        self.state.technical_summary = cleaned_data.get("technical_summary", "")
        self.state.city = cleaned_data.get("city", "Não foi possível determinar")
        self.state.opening_date = cleaned_data.get("opening_date", "")
        
        # Corrige o campo title se vier igual ao target
        title = cleaned_data.get("title", "")
        if title.strip().lower() == self.target.strip().lower():
            title = "Não disponível"
        
        # Atualiza o metadata
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
        system_content = """Você é um especialista em análise de editais de licitação.
        Sua tarefa é analisar se o edital é relevante para o target especificado.
        
        Considere:
        1. Sinônimos e termos relacionados
        2. Especificações técnicas que indiquem o tipo de equipamento
        3. Contexto do uso do equipamento/serviço
        
        Seja preciso e objetivo na sua análise."""

        user_content = f"""
        Target: {self.target}
        
        Resumo Técnico do Edital:
        {self.state.technical_summary}
        
        Informações Adicionais:
        - Objeto: {self.state.metadata.get('object', '')}
        - Especificações: {self.state.metadata.get('specifications', '')}
        - Quantidades: {self.state.metadata.get('quantities', '')}
        """
        
        response = call_ollama(system_content, user_content, temperature=0, max_tokens=1000, response_format=TargetAnalysis)
        response_obj = clean_and_parse_json(response, TargetAnalysis)
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

        system_content = """Você é um especialista em análise de quantidades em editais.
        Sua tarefa é identificar e somar todas as quantidades relevantes para o target.
        
        Instruções:
        1. Analise cuidadosamente o resumo do edital
        2. Identifique TODAS as quantidades mencionadas
        3. Some as quantidades relevantes para o target
        4. Especifique a unidade de medida
        5. Explique como chegou na quantidade total
        
        Seja preciso e detalhado na sua análise."""

        user_content = f"""
        Target: {self.target}
        Threshold: {self.threshold}
        
        Resumo Técnico do Edital:
        {self.state.technical_summary}
        """
        
        try:
            response = call_ollama(system_content, user_content, temperature=0, max_tokens=1000, response_format=QuantitiesAnalysis)
            response_obj = clean_and_parse_json(response, QuantitiesAnalysis)
            
            if response_obj.total_quantity >= self.threshold:
                self.state.threshold_match = "true"
            else:
                self.state.threshold_match = "false"
            
            self.state.metadata["quantities"] = f"{response_obj.total_quantity} {response_obj.unit} - {response_obj.explanation}"
            
        except Exception as e:
            self.state.threshold_match = "inconclusive"
            self.state.metadata["quantities"] = f"Erro ao processar quantidades: {str(e)}"

    @listen(check_threshold)
    @handle_flow_error
    def generate_justification(self):
        """Gera justificativa para a decisão."""
        system_content = """Você é um especialista em justificar decisões sobre editais.
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
        6. Use português claro e acessível
        
        FORMATO DE RESPOSTA:
        Responda APENAS com o texto da justificativa, sem formatação adicional.
        NÃO use JSON, markdown ou outros formatos."""

        user_content = f"""
        Target: {self.target}
        Target Match: {self.state.target_match}
        Threshold Match: {self.state.threshold_match}
        Threshold: {self.threshold}
        
        Resumo Executivo do Edital:
        {self.state.executive_summary}
        """
        
        response = call_ollama(system_content, user_content, temperature=0, max_tokens=200)
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
