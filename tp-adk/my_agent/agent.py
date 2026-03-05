import logging
import json
from google.adk.agents import LlmAgent, SequentialAgent, ParallelAgent
from google.adk.tools.agent_tool import AgentTool
from google.adk.agents.callback_context import CallbackContext
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.genai import types as genai_types
from google.genai import types as genai_types

from .tools.my_tools import (
    search_flights,
    estimate_flight_price,
    search_hotels,
    search_activities,
    calculate_budget,
    get_weather_forecast,
)

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════
# CALLBACKS (contrainte 6 : 2 callbacks de types différents)
# ═══════════════════════════════════════════════════════

def before_llm_callback(
    callback_context: CallbackContext,
    llm_request: LlmRequest,
) -> LlmRequest | None:
    """
    before_model callback : log chaque appel LLM et injecte un rappel
    de langue française dans le system prompt si absent.
    """
    agent_name = callback_context.agent_name
    logger.info(f"[CALLBACK] before_llm → agent='{agent_name}'")

    if llm_request.config and llm_request.config.system_instruction:
        current = llm_request.config.system_instruction
        if isinstance(current, str) and "français" not in current.lower():
            llm_request.config.system_instruction = (
                current + "\n\nRéponds toujours en français."
            )
    return None  


def after_agent_callback(
    callback_context: CallbackContext,
) -> genai_types.Content | None:
    """
    after_agent callback : log la fin d'exécution de chaque agent
    et sauvegarde un résumé dans l'état partagé.
    """
    agent_name = callback_context.agent_name
    logger.info(f"[CALLBACK] after_agent → agent='{agent_name}' terminé.")

    state = callback_context.state
    completed = state.get("completed_agents", [])
    if agent_name not in completed:
        completed.append(agent_name)
        state["completed_agents"] = completed

    return None  

def after_model_callback(
    callback_context: CallbackContext,
    llm_response: LlmResponse,
) -> LlmResponse | None:
    """
    after_model callback : nettoie les réponses JSON
    et extrait uniquement le texte lisible.
    """
    if not llm_response.content or not llm_response.content.parts:
        return None

    new_parts = []
    changed = False

    for part in llm_response.content.parts:
        if not hasattr(part, "text") or not part.text:
            new_parts.append(part)
            continue

        raw = part.text.strip()
        cleaned = raw

        try:
            data = json.loads(raw)
            # Cherche "content" en profondeur
            def find_content(d):
                if isinstance(d, dict):
                    if "content" in d and isinstance(d["content"], str):
                        return d["content"]
                    for v in d.values():
                        result = find_content(v)
                        if result:
                            return result
                return None

            found = find_content(data)
            if found:
                cleaned = found
                changed = True
        except (json.JSONDecodeError, TypeError):
            pass  

        new_parts.append(genai_types.Part(text=cleaned))

    if changed:
        return LlmResponse(
            content=genai_types.Content(
                role=llm_response.content.role,
                parts=new_parts,
            )
        )

    return None


# ═══════════════════════════════════════════════════════
# LEAF AGENTS (LlmAgent) — contrainte 1 : ≥ 3 LlmAgent
# ═══════════════════════════════════════════════════════

flight_agent = LlmAgent(
    name="flight_agent",
    model="ollama/mistral",
    description="Recherche des vols disponibles entre deux villes.",
    instruction="""Tu es un agent de recherche de vols. 
Appelle search_flights avec les paramètres de la conversation.
Ensuite, réponds UNIQUEMENT avec un texte en français lisible, sans JSON, sans accolades, sans guillemets techniques.
Format de réponse souhaité :
Vols disponibles :
- [Compagnie] : [durée]h — [prix]€ ([places] places)
""",
    tools=[search_flights, estimate_flight_price],
    output_key="flight_results",
    before_model_callback=before_llm_callback,
    after_model_callback=after_model_callback,
    after_agent_callback=after_agent_callback,
)

hotel_agent = LlmAgent(
    name="hotel_agent",
    model="ollama/mistral",
    description="Recherche des hôtels dans la ville de destination.",
    instruction="""Tu es un agent de recherche d'hôtels.
Appelle search_hotels avec les paramètres de la conversation.
Ensuite, réponds UNIQUEMENT avec un texte en français lisible, sans JSON, sans accolades, sans guillemets techniques.
Format de réponse souhaité :
Hôtels disponibles :
- [Nom] ([étoiles]) : [prix]€/nuit — Note [note]/10
""",
    tools=[search_hotels],
    output_key="hotel_results",
    before_model_callback=before_llm_callback,
    after_model_callback=after_model_callback,
    after_agent_callback=after_agent_callback,
)

activities_agent = LlmAgent(
    name="activities_agent",
    model="ollama/mistral",
    description="Trouve des activités touristiques.",
    instruction="""Tu es un guide touristique.
Appelle search_activities avec le paramètre city de la conversation.
Ensuite, réponds UNIQUEMENT avec un texte en français lisible, sans JSON, sans accolades, sans guillemets techniques.
Format de réponse souhaité :
Activités recommandées :
- [Nom] : [durée]h — [prix]€
""",
    tools=[search_activities],
    output_key="activities_results",
    before_model_callback=before_llm_callback,
    after_model_callback=after_model_callback,
    after_agent_callback=after_agent_callback,
)

budget_agent = LlmAgent(
    name="budget_agent",
    model="ollama/mistral",
    description="Calcule le budget total du voyage.",
    instruction="""Tu es un conseiller financier voyage.
Appelle calculate_budget(flight_price=400.0, hotel_price_per_night=120.0, num_nights=3, activities_budget=100.0, daily_food_budget=50.0).
Ensuite, réponds UNIQUEMENT avec un texte en français lisible, sans JSON, sans accolades, sans guillemets techniques.
Format de réponse souhaité :
Budget estimé :
- Vols : [montant]€
- Hôtel : [montant]€
- Activités : [montant]€
- Repas : [montant]€
- TOTAL : [montant]€
""",
    tools=[calculate_budget],
    output_key="budget_summary",
    before_model_callback=before_llm_callback,
    after_model_callback=after_model_callback,
    after_agent_callback=after_agent_callback,
)

weather_agent = LlmAgent(
    name="weather_agent",
    model="ollama/mistral",
    description="Donne la météo pour la destination.",
    instruction="""Tu es un météorologue voyage.
Appelle get_weather_forecast avec les paramètres city et date de la conversation.
Ensuite, réponds UNIQUEMENT avec un texte en français lisible, sans JSON, sans accolades, sans guillemets techniques.
Format de réponse souhaité :
Météo prévue :
- Condition : [condition]
- Température : [temp]°C
- Humidité : [humidité]%
- Conseil : [conseil vestimentaire]
""",
    tools=[get_weather_forecast],
    output_key="weather_info",
    before_model_callback=before_llm_callback,
    after_model_callback=after_model_callback,
    after_agent_callback=after_agent_callback,
)

# ═══════════════════════════════════════════════════════
# WORKFLOW AGENTS
# Contrainte 3 : 2 types différents de workflow agents
# ═══════════════════════════════════════════════════════

# --- Workflow 1 : SequentialAgent ---
# Vols → Hôtels → Activités (ordre important : chaque étape dépend de la précédente)
planner_agent = SequentialAgent(
    name="planner_agent",
    description="Planifie le voyage étape par étape : vols, hôtels, activités.",
    sub_agents=[flight_agent, hotel_agent, activities_agent],
)

# --- Workflow 2 : ParallelAgent ---
# Budget + Météo peuvent être calculés en parallèle car indépendants
parallel_info_agent = ParallelAgent(
    name="parallel_info_agent",
    description="Récupère en parallèle le budget et la météo du voyage.",
    sub_agents=[budget_agent, weather_agent],
)


# ═══════════════════════════════════════════════════════
# ROOT AGENT
# Contrainte 5 : transfer_to_agent + AgentTool
# ═══════════════════════════════════════════════════════

root_agent = LlmAgent(
    name="travel_assistant",
    model="ollama/mistral",
    description="Assistant de voyage principal.",
    instruction="""
Tu es un assistant de voyage intelligent. Réponds toujours en français.

Quand l'utilisateur demande une planification de voyage :
1. Extrais destination, ville de départ, dates et nombre de voyageurs depuis son message.
2. Délègue au planner_agent pour vols + hôtels + activités.
3. Synthétise les résultats en un récapitulatif clair.

Si des informations manquent (destination, dates), demande-les poliment.
Ne mentionne jamais les noms techniques des agents internes.
""",
    tools=[
        AgentTool(agent=budget_agent),
        AgentTool(agent=weather_agent),
    ],
    sub_agents=[planner_agent, parallel_info_agent],
    output_key="final_travel_plan",
    after_model_callback=after_model_callback,
    after_agent_callback=after_agent_callback,
)