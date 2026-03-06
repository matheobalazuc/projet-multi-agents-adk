import logging
import json
import re
import os
from dotenv import load_dotenv
from google.adk.agents import LlmAgent, SequentialAgent, ParallelAgent
from google.adk.tools.agent_tool import AgentTool
from google.adk.agents.callback_context import CallbackContext
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.genai import types as genai_types

from .tools.my_tools import (
    search_flights,
    estimate_flight_price,
    search_hotels,
    search_activities,
    calculate_budget,
    get_weather_forecast,
)

load_dotenv()
MODEL = os.getenv("ADK_MODEL_NAME", "ollama/mistral")

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════
# OUTILS AUTORISES
# ═══════════════════════════════════════════════════════

OUTILS_AUTORISES = {
    "search_flights", "estimate_flight_price",
    "search_hotels", "search_activities",
    "calculate_budget", "get_weather_forecast",
    "budget_agent", "weather_agent",
    "transfer_to_agent",
    "planner_agent", "parallel_info_agent",
}

PHRASES_METEO  = ["meteo", "quel temps", "temperature", "climat", "previsions", "weather"]
PHRASES_BUDGET = ["budget", "combien", "quel prix", "tarif", "cout", "prix total"]
PHRASES_PLANIF = ["planifie", "organise", "je veux aller", "je voudrais aller",
                  "partir a", "partir pour", "voyage de", "sejour a", "trip a"]


# ═══════════════════════════════════════════════════════
# UTILITAIRES D EXTRACTION
# ═══════════════════════════════════════════════════════

def detecter_intention(message: str) -> str | None:
    msg = message.lower()
    meteo  = any(p in msg for p in PHRASES_METEO)
    budget = any(p in msg for p in PHRASES_BUDGET)
    planif = any(p in msg for p in PHRASES_PLANIF)
    if meteo and budget:      return "parallel_info_agent"
    if meteo and not planif:  return "weather_agent"
    if budget and not planif: return "budget_agent"
    if planif:                return "planner_agent"
    return None


def extraire_dernier_message_user(llm_request: LlmRequest) -> str:
    for content in reversed(llm_request.contents or []):
        if content.role == "user":
            for part in content.parts or []:
                if hasattr(part, "text") and part.text:
                    return part.text
    return ""


def extraire_ville(texte: str) -> str:
    mots = texte.lower().split()
    prepositions = {"a", "pour", "vers", "en", "au", "aux", "depuis"}
    for i, mot in enumerate(mots):
        if mot in prepositions and i + 1 < len(mots):
            ville = mots[i + 1].strip(".,?!")
            if len(ville) > 2:
                return ville.capitalize()
    for mot in reversed(mots):
        mot = mot.strip(".,?!")
        if len(mot) > 3 and mot not in prepositions:
            return mot.capitalize()
    return "Paris"


def extraire_origine(texte: str) -> str:
    mots = texte.lower().split()
    for i, mot in enumerate(mots):
        if mot in {"depuis", "de"} and i + 1 < len(mots):
            ville = mots[i + 1].strip(".,?!")
            if len(ville) > 2:
                return ville.capitalize()
    return "Paris"


def extraire_nuits(texte: str) -> int:
    match = re.search(r'(\d+)\s*nuits?', texte.lower())
    if match:
        return int(match.group(1))
    match = re.search(r'(\d+)\s*jours?', texte.lower())
    if match:
        return max(1, int(match.group(1)) - 1)
    return 3


# ═══════════════════════════════════════════════════════
# FORMATAGE DES REPONSES
# ═══════════════════════════════════════════════════════

def formater_meteo(data: dict, ville: str) -> str:
    if data.get("status") != "success":
        return f"Impossible de recuperer la meteo pour {ville}."
    return "\n".join([
        f"Meteo a {ville}",
        "-" * 32,
        f"Condition    : {data.get('condition', 'N/A')}",
        f"Temperature  : {data.get('temperature_celsius', 'N/A')} degres C",
        f"Humidite     : {data.get('humidity_percent', 'N/A')} pourcent",
        f"Vent         : {data.get('wind_kmh', 'N/A')} km/h",
    ])


def formater_budget(data: dict, num_nights: int) -> str:
    if data.get("status") != "success":
        return "Impossible de calculer le budget."
    b = data.get("breakdown", {})
    return "\n".join([
        "Budget estime du voyage",
        "-" * 32,
        f"Vols        : {b.get('flights_eur', 0):.0f} euros",
        f"Hotel       : {b.get('hotel_eur', 0):.0f} euros ({num_nights} nuits)",
        f"Activites   : {b.get('activities_eur', 0):.0f} euros",
        f"Repas       : {b.get('food_eur', 0):.0f} euros",
        "-" * 32,
        f"Total       : {data.get('total_eur', 0):.0f} euros",
    ])


def formater_vols(data: dict) -> str:
    if data.get("status") != "success":
        return "Aucun vol trouve."
    lignes = ["Vols disponibles", "-" * 32]
    for v in data.get("flights", []):
        lignes.append(
            f"- {v['airline']} : {v['duration_hours']}h de vol  |  "
            f"{v['price_eur']:.0f} euros  |  {v['seats_available']} places"
        )
    return "\n".join(lignes)


def formater_hotels(data: dict) -> str:
    if data.get("status") != "success":
        return "Aucun hotel trouve."
    lignes = ["Hotels disponibles", "-" * 32]
    for h in data.get("hotels", []):
        lignes.append(
            f"- {h['name']} : {h['stars']} etoiles  |  "
            f"{h['price_per_night_eur']:.0f} euros/nuit  |  note {h['rating']}/10"
        )
    return "\n".join(lignes)


def formater_activites(data: dict) -> str:
    if data.get("status") != "success":
        return "Aucune activite trouvee."
    lignes = ["Activites recommandees", "-" * 32]
    for a in data.get("activities", []):
        prix = f"{a['price_eur']} euros" if a["price_eur"] > 0 else "gratuit"
        lignes.append(f"- {a['name']} : {a['duration_hours']}h  |  {prix}")
    return "\n".join(lignes)


def formater_recapitulatif(state: dict, destination: str) -> str:
    """
    Assemble le recapitulatif complet depuis le state partage.
    Appele par le root_agent apres planner_agent.
    """
    sections = [
        f"Recapitulatif de votre voyage a {destination}",
        "=" * 40,
        state.get("flight_results", "Vols : non disponible"),
        "",
        state.get("hotel_results", "Hotels : non disponible"),
        "",
        state.get("activities_results", "Activites : non disponible"),
        "=" * 40,
        "Bon voyage !",
    ]
    return "\n".join(sections)


# ═══════════════════════════════════════════════════════
# CALLBACKS
# before_model_callback + after_agent_callback
# ═══════════════════════════════════════════════════════

def before_llm_callback(
    callback_context: CallbackContext,
    llm_request: LlmRequest,
) -> LlmResponse | None:
    """
    before_model callback :
    - Log chaque appel LLM
    - Leaf agents : appel direct de l outil, formatage texte, retour LlmResponse
    - Root agent : supprime outils inventes, detecte boucles, injecte routage
    """
    agent_name = callback_context.agent_name
    logger.info(f"[CALLBACK] before_llm -> agent='{agent_name}'")

    state = callback_context.state

    # Extrait et stocke destination / origine / nuits si absent du state
    destination = state.get("destination", "")
    origine     = state.get("origin", "Paris")
    if not destination:
        dernier     = extraire_dernier_message_user(llm_request)
        destination = extraire_ville(dernier)
        origine     = extraire_origine(dernier)
        num_nights  = extraire_nuits(dernier)
        callback_context.state["destination"] = destination
        callback_context.state["origin"]      = origine
        callback_context.state["num_nights"]  = num_nights
        logger.info(f"[STATE] destination={destination} | origine={origine} | nuits={num_nights}")

    # ── LEAF AGENTS ────────────────────────────────────────────────────────────

    if agent_name == "flight_agent":
        logger.info(f"[CALLBACK] flight_agent : {origine} -> {destination}")
        data  = search_flights(origin=origine, destination=destination,
                               date="2025-08-01", passengers=1)
        texte = formater_vols(data)
        callback_context.state["flight_results"] = texte
        return LlmResponse(content=genai_types.Content(
            role="model", parts=[genai_types.Part(text=texte)]))

    if agent_name == "hotel_agent":
        logger.info(f"[CALLBACK] hotel_agent : {destination}")
        data  = search_hotels(city=destination, check_in="2025-08-01",
                              check_out="2025-08-05", guests=1)
        texte = formater_hotels(data)
        callback_context.state["hotel_results"] = texte
        return LlmResponse(content=genai_types.Content(
            role="model", parts=[genai_types.Part(text=texte)]))

    if agent_name == "activities_agent":
        logger.info(f"[CALLBACK] activities_agent : {destination}")
        data  = search_activities(city=destination)
        texte = formater_activites(data)
        callback_context.state["activities_results"] = texte
        return LlmResponse(content=genai_types.Content(
            role="model", parts=[genai_types.Part(text=texte)]))

    if agent_name == "budget_agent":
        num_nights = int(state.get("num_nights", 3))
        logger.info(f"[CALLBACK] budget_agent : {num_nights} nuits")
        data  = calculate_budget(flight_price=400.0, hotel_price_per_night=120.0,
                                  num_nights=num_nights, activities_budget=100.0,
                                  daily_food_budget=50.0)
        texte = formater_budget(data, num_nights)
        callback_context.state["budget_summary"] = texte
        return LlmResponse(content=genai_types.Content(
            role="model", parts=[genai_types.Part(text=texte)]))

    if agent_name == "weather_agent":
        logger.info(f"[CALLBACK] weather_agent : {destination}")
        data  = get_weather_forecast(city=destination, date="2025-08-01")
        texte = formater_meteo(data, destination)
        callback_context.state["weather_info"] = texte
        return LlmResponse(content=genai_types.Content(
            role="model", parts=[genai_types.Part(text=texte)]))

    # ── ROOT AGENT ─────────────────────────────────────────────────────────────

    if agent_name == "travel_assistant":

        # Supprime les outils inventes par le LLM
        if llm_request.config and hasattr(llm_request.config, "tools") and llm_request.config.tools:
            filtres = []
            for tool in llm_request.config.tools:
                nom = getattr(tool, "name", None)
                if nom and nom not in OUTILS_AUTORISES:
                    logger.warning(f"[CALLBACK] outil invente supprime : '{nom}'")
                    continue
                filtres.append(tool)
            llm_request.config.tools = filtres

        # Detection boucle : meme outil appele 2 fois -> force sortie texte
        tool_call_count = 0
        last_tool = None
        for content in (llm_request.contents or []):
            if content.role == "model":
                for part in content.parts or []:
                    if hasattr(part, "function_call") and part.function_call:
                        nom = getattr(part.function_call, "name", "")
                        if nom == last_tool:
                            tool_call_count += 1
                        else:
                            last_tool = nom
                            tool_call_count = 1

        if tool_call_count >= 2:
            logger.warning(f"[CALLBACK] boucle detectee sur '{last_tool}' -> force reponse finale")
            # Assemble directement le recapitulatif depuis le state sans repasser par le LLM
            destination = state.get("destination", "votre destination")
            texte_final = formater_recapitulatif(state, destination)
            return LlmResponse(content=genai_types.Content(
                role="model", parts=[genai_types.Part(text=texte_final)]))

        # Routage force vers le bon agent selon l intention detectee
        dernier_message = extraire_dernier_message_user(llm_request)
        if dernier_message:
            cible = detecter_intention(dernier_message)
            if cible and llm_request.config:
                logger.info(f"[CALLBACK] routage force -> {cible}")
                llm_request.config.system_instruction = (
                    (llm_request.config.system_instruction or "")
                    + f"\n\nIMPERATIF : appelle transfer_to_agent avec name='{cible}'."
                      " Une seule fois. Apres le resultat, redige ta reponse finale en texte."
                )

    return None


def after_agent_callback(
    callback_context: CallbackContext,
) -> genai_types.Content | None:
    """
    after_agent callback :
    - Log la fin d execution de chaque agent
    - Met a jour completed_agents dans le state partage (contrainte 4)
    """
    agent_name = callback_context.agent_name
    logger.info(f"[CALLBACK] after_agent -> '{agent_name}' termine.")
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
    after_model callback :
    - Neutralise les function_calls inventes par le LLM
    - Nettoie les reponses JSON residuelles pour retourner du texte pur
    """
    if not llm_response.content or not llm_response.content.parts:
        return None

    new_parts = []
    changed   = False

    for part in llm_response.content.parts:

        # Neutralise les function_calls inventes
        if hasattr(part, "function_call") and part.function_call:
            nom = getattr(part.function_call, "name", "")
            if nom not in OUTILS_AUTORISES:
                logger.warning(f"[CALLBACK] function_call invente neutralise : '{nom}'")
                args  = getattr(part.function_call, "args", {})
                texte = ""
                for key in ["text", "content", "response", "message", "result"]:
                    val = args.get(key)
                    if val and isinstance(val, str):
                        texte = val
                        break
                if texte:
                    new_parts.append(genai_types.Part(text=texte))
                changed = True
                continue

        if not hasattr(part, "text") or not part.text:
            new_parts.append(part)
            continue

        raw     = part.text.strip()
        cleaned = raw

        # Extrait le texte lisible si la reponse contient du JSON
        try:
            data = json.loads(raw)
            def find_content(d):
                if isinstance(d, dict):
                    for key in ["content", "text", "message", "response"]:
                        if key in d and isinstance(d[key], str):
                            return d[key]
                    for v in d.values():
                        r = find_content(v)
                        if r:
                            return r
                return None
            found = find_content(data)
            if found:
                cleaned = found
                changed = True
        except (json.JSONDecodeError, TypeError):
            pass

        new_parts.append(genai_types.Part(text=cleaned))

    if not changed:
        return None
    if not any(getattr(p, "text", "").strip() for p in new_parts if hasattr(p, "text")):
        return None

    return LlmResponse(
        content=genai_types.Content(
            role=llm_response.content.role,
            parts=new_parts,
        )
    )


# ═══════════════════════════════════════════════════════
# LEAF AGENTS
# ═══════════════════════════════════════════════════════

flight_agent = LlmAgent(
    name="flight_agent",
    model=MODEL,
    description="Recherche des vols disponibles entre deux villes.",
    instruction="Recherche et retourne les vols disponibles.",
    tools=[search_flights, estimate_flight_price],
    output_key="flight_results",
    before_model_callback=before_llm_callback,
    after_model_callback=after_model_callback,
    after_agent_callback=after_agent_callback,
)

hotel_agent = LlmAgent(
    name="hotel_agent",
    model=MODEL,
    description="Recherche des hotels dans la ville de destination.",
    instruction="Recherche et retourne les hotels disponibles.",
    tools=[search_hotels],
    output_key="hotel_results",
    before_model_callback=before_llm_callback,
    after_model_callback=after_model_callback,
    after_agent_callback=after_agent_callback,
)

activities_agent = LlmAgent(
    name="activities_agent",
    model=MODEL,
    description="Trouve des activites touristiques a faire sur place.",
    instruction="Recherche et retourne les activites disponibles.",
    tools=[search_activities],
    output_key="activities_results",
    before_model_callback=before_llm_callback,
    after_model_callback=after_model_callback,
    after_agent_callback=after_agent_callback,
)

budget_agent = LlmAgent(
    name="budget_agent",
    model=MODEL,
    description="Calcule le budget total estime d un voyage.",
    instruction="Calcule et retourne le budget estime du voyage.",
    tools=[calculate_budget],
    output_key="budget_summary",
    before_model_callback=before_llm_callback,
    after_model_callback=after_model_callback,
    after_agent_callback=after_agent_callback,
)

weather_agent = LlmAgent(
    name="weather_agent",
    model=MODEL,
    description="Fournit les previsions meteo pour une destination.",
    instruction="Recupere et retourne les previsions meteo.",
    tools=[get_weather_forecast],
    output_key="weather_info",
    before_model_callback=before_llm_callback,
    after_model_callback=after_model_callback,
    after_agent_callback=after_agent_callback,
)


# ═══════════════════════════════════════════════════════
# WORKFLOW AGENTS
# ═══════════════════════════════════════════════════════

# SequentialAgent : vols -> hotels -> activites dans l ordre strict
planner_agent = SequentialAgent(
    name="planner_agent",
    description="Planifie le voyage complet : vols puis hotels puis activites dans l ordre.",
    sub_agents=[flight_agent, hotel_agent, activities_agent],
)

# ParallelAgent : budget et meteo recuperes simultanement car independants
parallel_info_agent = ParallelAgent(
    name="parallel_info_agent",
    description="Recupere budget et meteo en parallele simultanement.",
    sub_agents=[budget_agent, weather_agent],
)


# ═══════════════════════════════════════════════════════
# ROOT AGENT
# ═══════════════════════════════════════════════════════

root_agent = LlmAgent(
    name="travel_assistant",
    model=MODEL,
    description="Assistant de voyage principal qui orchestre et synthetise les resultats.",
    instruction="""Tu es un assistant de voyage. Tu reponds toujours en francais.
Les resultats des agents sont deja formates en texte dans le state.
Ne genere pas de JSON. Presente le texte tel quel, sans reformatage.

Outils directs :
- get_weather_forecast(city, date)
- calculate_budget(flight_price, hotel_price_per_night, num_nights, activities_budget, daily_food_budget)
- budget_agent
- weather_agent

Sous-agents via transfer_to_agent :
- planner_agent       : vols + hotels + activites
- parallel_info_agent : budget + meteo en parallele

Interdit : flight_agent, hotel_agent, activities_agent, response, format_response, generate_response

Apres UN appel, redige IMMEDIATEMENT ta reponse finale. Tu n appelles plus rien.

Regle 1 - METEO :
Appelle get_weather_forecast(city="...", date="2025-08-01")
Reponse avec les donnees recues :
  Meteo a [ville]
  --------------------------------
  Condition    : [valeur]
  Temperature  : [valeur] degres C
  Humidite     : [valeur] pourcent
  Vent         : [valeur] km/h

Regle 2 - BUDGET :
Appelle calculate_budget(flight_price=400.0, hotel_price_per_night=120.0, num_nights=X, activities_budget=100.0, daily_food_budget=50.0)
Reponse avec les donnees recues :
  Budget estime du voyage
  --------------------------------
  Vols        : [valeur] euros
  Hotel       : [valeur] euros
  Activites   : [valeur] euros
  Repas       : [valeur] euros
  --------------------------------
  Total       : [valeur] euros

Regle 3 - PLANIFICATION :
Appelle transfer_to_agent avec name="planner_agent"
Quand tu recois les resultats, presente le contenu de flight_results,
hotel_results et activities_results dans cet ordre :
  Recapitulatif de votre voyage a [ville]
  ========================================
  [flight_results]

  [hotel_results]

  [activities_results]
  ========================================
  Bon voyage !

Regle 4 - BUDGET + METEO :
Appelle transfer_to_agent avec name="parallel_info_agent"
Presente budget_summary puis weather_info l un apres l autre.

Regle 5 - Infos manquantes : pose une seule question precise.
""",
    tools=[
        get_weather_forecast,
        calculate_budget,
        AgentTool(agent=budget_agent),
        AgentTool(agent=weather_agent),
    ],
    sub_agents=[planner_agent, parallel_info_agent],
    output_key="final_travel_plan",
    before_model_callback=before_llm_callback,
    after_model_callback=after_model_callback,
    after_agent_callback=after_agent_callback,
)