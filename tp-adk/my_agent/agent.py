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

PHRASES_METEO = [
    "meteo","météo",
    "quel temps","temps a","temps à",
    "temperature","température",
    "climat",
    "previsions","prévisions",
    "il fait combien",
    "weather"
]

PHRASES_BUDGET = [
    "budget",
    "combien",
    "quel prix",
    "tarif",
    "cout","coût",
    "prix total",
    "combien ca coute","combien ça coûte",
]

PHRASES_PLANIF = [
    "planifie","planifier",
    "organise","organiser",
    "je veux aller",
    "je voudrais aller",
    "je souhaite aller",
    "partir a","partir à",
    "partir pour",
    "voyage a","voyage à",
    "voyage de",
    "sejour a","séjour à",
    "vacances a","vacances à",
    "trip a","trip à",
]

MOIS = {
    "janvier","fevrier","février","mars","avril","mai","juin",
    "juillet","aout","août","septembre","octobre","novembre","decembre","décembre"
}


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
    # FIX Bug 3 : planification → on lance planner + parallel_info en séquence
    # On retourne planner_agent ici ; le root_agent appellera aussi parallel_info_agent
    if planif:                return "planner_agent"
    return None


def extraire_dernier_message_user(llm_request: LlmRequest) -> str:
    """
    Extrait le VRAI dernier message utilisateur depuis llm_request.contents.
    On ignore les messages dont le rôle n'est pas 'user' et on s'assure
    que le texte extrait ne ressemble pas à un prompt système injecté par ADK.
    """
    for content in reversed(llm_request.contents or []):
        if content.role == "user":
            for part in content.parts or []:
                if hasattr(part, "text") and part.text:
                    texte = part.text.strip()
                    # FIX Bug 1 : on ignore les blocs qui commencent par "Context:"
                    # car ce sont des injections internes d'ADK, pas des messages user
                    if texte.lower().startswith("context:"):
                        continue
                    return texte
    return ""


def extraire_ville(texte: str) -> str | None:
    mots = texte.lower().split()
    prepositions = {"a", "à", "pour", "vers", "en", "au", "aux"}

    for i, mot in enumerate(mots):
        if mot in prepositions and i + 1 < len(mots):
            ville = mots[i + 1].strip(".,?!")

            if ville in MOIS:
                continue

            if len(ville) > 2:
                return ville.capitalize()

    for mot in reversed(mots):
        mot = mot.strip(".,?!")

        if mot in MOIS:
            continue

        if len(mot) > 3 and mot not in prepositions:
            return mot.capitalize()

    return None


def extraire_origine(texte: str) -> str | None:
    mots = texte.lower().split()

    for i, mot in enumerate(mots):
        if mot in {"depuis", "de"} and i + 1 < len(mots):
            ville = mots[i + 1].strip(".,?!")

            if len(ville) > 2:
                return ville.capitalize()

    return None


def extraire_nuits(texte: str) -> int | None:
    match = re.search(r'(\d+)\s*nuits?', texte.lower())
    if match:
        return int(match.group(1))

    match = re.search(r'(\d+)\s*jours?', texte.lower())
    if match:
        return max(1, int(match.group(1)) - 1)

    return None


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


def formater_recapitulatif_complet(state: dict, destination: str) -> str:
    """
    FIX Bug 3 : Recapitulatif complet avec budget ET meteo en plus
    des vols, hotels et activites quand on planifie un voyage.
    """
    sections = [
        f"Recapitulatif de votre voyage a {destination}",
        "=" * 40,
        state.get("flight_results", "Vols : non disponible"),
        "",
        state.get("hotel_results", "Hotels : non disponible"),
        "",
        state.get("activities_results", "Activites : non disponible"),
    ]

    # Ajout budget et meteo si disponibles
    if state.get("budget_summary"):
        sections += ["", state["budget_summary"]]
    if state.get("weather_info"):
        sections += ["", state["weather_info"]]

    sections += ["=" * 40, "Bon voyage !"]
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

    CORRECTIONS :
    - Bug 1 : extraire_dernier_message_user ignore les blocs "Context:" d'ADK,
              ce qui evitait que destination = "Context:"
    - Bug 2 : la detection de boucle se base desormais uniquement sur les
              appels LLM du TOUR COURANT (depuis le dernier message user),
              evitant que les appels passes bloquent les nouvelles requetes.
    - Bug 3 : lors d une planification, on lance aussi parallel_info_agent
              apres planner_agent pour avoir budget + meteo dans le recap.
    """
    agent_name = callback_context.agent_name
    logger.info(f"[CALLBACK] before_llm -> agent='{agent_name}'")

    state = callback_context.state

    # FIX BUG 1 (vraie correction) :
    # Les leaf agents (flight, hotel, activities, budget, weather) reçoivent un message
    # reconstruit par ADK qui commence par "Context: ..." — PAS le message utilisateur original.
    # Si on tente d'extraire la ville depuis ce contenu, on obtient "Context" comme destination.
    #
    # Règle : seul le root_agent (travel_assistant) extrait et écrit dans le state.
    # Les leaf agents lisent UNIQUEMENT le state, jamais llm_request.
    #
    # On détecte si on est un leaf agent pour ne PAS faire l'extraction.
    LEAF_AGENTS = {"flight_agent", "hotel_agent", "activities_agent", "budget_agent", "weather_agent"}

    if agent_name not in LEAF_AGENTS:
        # Extraction uniquement pour le root_agent
        dernier = extraire_dernier_message_user(llm_request)

        destination_extrait = extraire_ville(dernier) if dernier else None
        origine_extrait     = extraire_origine(dernier) if dernier else None
        nuits_extrait       = extraire_nuits(dernier) if dernier else None

        # On écrase uniquement si on a extrait quelque chose de valide
        if destination_extrait:
            state["destination"] = destination_extrait
        if origine_extrait:
            state["origin"] = origine_extrait
        if nuits_extrait:
            state["num_nights"] = nuits_extrait

    # Lecture du state (valeurs déjà correctes pour les leaf agents)
    destination = state.get("destination", "")
    origine     = state.get("origin", "Paris")
    num_nights  = state.get("num_nights", 3)

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
        try:
            data = get_weather_forecast(city=destination, date="2025-08-01")
        except Exception as e:
            logger.error(f"[CALLBACK] weather_agent erreur : {e}")
            data = {"status": "error"}
        # data peut être None si l'outil retourne None
        if not data or not isinstance(data, dict):
            data = {"status": "error"}
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

        # FIX Bug 2 : detection de boucle uniquement sur le TOUR COURANT
        # On compte les appels depuis le dernier message utilisateur,
        # pas sur l'ensemble de l'historique de la session.
        tool_call_count = 0
        last_tool = None
        in_current_turn = False

        for content in (llm_request.contents or []):
            # Dès qu'on voit le dernier message user, on commence à compter
            if content.role == "user":
                # Vérifie si c'est bien le dernier message user (le message courant)
                for part in content.parts or []:
                    if hasattr(part, "text") and part.text:
                        texte_part = part.text.strip()
                        if not texte_part.lower().startswith("context:"):
                            in_current_turn = True
                            tool_call_count = 0  # reset au nouveau tour
                            last_tool = None

            if in_current_turn and content.role == "model":
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

                # FIX Bug 3 : si c'est une planification, on demande au LLM
                # d'appeler AUSSI parallel_info_agent apres planner_agent
                if cible == "planner_agent":
                    extra_instruction = (
                        f"\n\nIMPERATIF : appelle d'abord transfer_to_agent avec name='planner_agent'."
                        " Attends le résultat. Ensuite appelle transfer_to_agent avec"
                        " name='parallel_info_agent' pour récupérer budget et météo."
                        " Après les deux résultats, rédige ta réponse finale en texte avec le récapitulatif complet."
                    )
                else:
                    extra_instruction = (
                        f"\n\nIMPERATIF : appelle transfer_to_agent avec name='{cible}'."
                        " Une seule fois. Apres le resultat, redige ta reponse finale en texte."
                    )

                llm_request.config.system_instruction = (
                    (llm_request.config.system_instruction or "")
                    + extra_instruction
                )

    return None


def after_agent_callback(
    callback_context: CallbackContext,
) -> genai_types.Content | None:
    """
    after_agent callback :
    - Log la fin d execution de chaque agent
    - Met a jour completed_agents dans le state partage (contrainte 4)

    FIX Bug 3 : si planner_agent vient de finir, on assemble un recap complet
    incluant budget et météo si disponibles dans le state.
    """
    agent_name = callback_context.agent_name
    logger.info(f"[CALLBACK] after_agent -> '{agent_name}' termine.")
    state = callback_context.state
    completed = state.get("completed_agents", [])
    if agent_name not in completed:
        completed.append(agent_name)
        state["completed_agents"] = completed

    # FIX Bug 3 : après parallel_info_agent, si planner_agent a déjà tourné,
    # on met à jour final_travel_plan avec le recap complet
    if agent_name == "parallel_info_agent" and "planner_agent" in completed:
        destination = state.get("destination", "votre destination")
        texte_final = formater_recapitulatif_complet(state, destination)
        state["final_travel_plan"] = texte_final
        logger.info("[CALLBACK] recap complet (vols+hotels+activites+budget+meteo) mis a jour dans state")

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
                # args peut être None si le LLM n'a pas fourni d'arguments
                args  = getattr(part.function_call, "args", {}) or {}
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

Regle 3 - PLANIFICATION COMPLETE :
1. Appelle transfer_to_agent avec name="planner_agent"
2. Ensuite appelle transfer_to_agent avec name="parallel_info_agent"
3. Presente le recap complet avec vols, hotels, activites, budget et meteo.

Regle 4 - BUDGET + METEO seulement :
Appelle transfer_to_agent avec name="parallel_info_agent"
Presente budget_summary puis weather_info l un apres l autre.

Regle 5 - Infos manquantes : pose une seule question precise.

APRES CHAQUE APPEL D AGENT : redige ta reponse finale en texte pur. Ne rappelle plus rien.
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