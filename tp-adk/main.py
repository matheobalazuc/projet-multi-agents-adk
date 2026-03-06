import asyncio
import logging
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types as genai_types

from my_agent.agent import root_agent

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


async def run_travel_assistant():

    # Initialisation du service de session en mémoire
    session_service = InMemorySessionService()

    # Création d'une session avec l'état initial partagé
    initial_state = {
        "destination": "",
        "origin": "",
        "travel_dates": "",
        "passengers": "",
        "num_nights": "",
        "activities_budget": 0.0,
        "completed_agents": [],
    }

    session = await session_service.create_session(
        app_name="travel_assistant",
        user_id="user_001",
        state=initial_state,
    )

    # Création du Runner
    runner = Runner(
        agent=root_agent,
        app_name="travel_assistant",
        session_service=session_service,
    )

    print("\n Bienvenue dans votre Assistant de Voyage!")
    print("=" * 55)
    print("Exemples de requêtes :")
    print("  - Je veux aller 4 nuits à Tokyo depuis Paris pour 2 personnes.")
    print("  - Planifie un voyage à Rome depuis Nice pour 3 nuits en août.")
    print("  - Quelle météo à Barcelone en septembre ?")
    print("  - Combien coûte un séjour à Lisbonne et quelle météo en juillet ?")
    print("=" * 55)
    print("Tapez 'quit' pour quitter.\n")

    while True:
        user_input = input("Vous : ").strip()
        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit", "q"):
            print("Au revoir ! Bon voyage !")
            break

        # Construction du message utilisateur
        message = genai_types.Content(
            role="user",
            parts=[genai_types.Part(text=user_input)],
        )

        print("\nAssistant : ", end="", flush=True)

        try:
            # On collecte le texte de tous les events et on n'affiche
            reponses_finales = []

            async for event in runner.run_async(
                user_id="user_001",
                session_id=session.id,
                new_message=message,
            ):
                # On ne garde que les events du root_agent avec du texte
                # is_final_response() indique la réponse définitive de l'agent courant
                if hasattr(event, "is_final_response") and event.is_final_response():
                    if event.content and event.content.parts:
                        texte_event = "".join(
                            p.text for p in event.content.parts
                            if hasattr(p, "text") and p.text
                        )
                        if texte_event.strip():
                            reponses_finales.append(texte_event)

            # Affiche uniquement la dernière réponse finale (évite les doublons)
            if reponses_finales:
                print(reponses_finales[-1], end="", flush=True)
            else:
                # Fallback : streaming classique si aucun event final détecté
                async for event in runner.run_async(
                    user_id="user_001",
                    session_id=session.id,
                    new_message=message,
                ):
                    if event.content and event.content.parts:
                        for part in event.content.parts:
                            if hasattr(part, "text") and part.text:
                                print(part.text, end="", flush=True)

        except Exception as e:
            logger.error(f"Erreur lors de l'exécution : {e}", exc_info=True)
            print(f"\n[Erreur] {e}")

        print("\n")


if __name__ == "__main__":
    asyncio.run(run_travel_assistant())