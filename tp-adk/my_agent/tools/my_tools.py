"""
Custom tools for the travel assistant multi-agent system.
"""

from typing import Optional
import random


# ─────────────────────────────────────────────
# FLIGHTS TOOLS
# ─────────────────────────────────────────────

def search_flights(
    origin: str,
    destination: str,
    date: str,
    passengers: int = 1
) -> dict:
    """
    Search for available flights between two cities on a given date.

    Args:
        origin: Departure city or airport code (e.g. "Paris", "CDG").
        destination: Arrival city or airport code (e.g. "Tokyo", "TYO").
        date: Travel date in YYYY-MM-DD format.
        passengers: Number of passengers (default 1).

    Returns:
        A dict with a list of available flights and their prices.
    """
    try:
        if not origin or not destination or not date:
            return {"error": "origin, destination and date are required."}

        # Simulated flight data
        airlines = ["AirFrance", "Lufthansa", "Emirates", "Japan Airlines"]
        flights = []
        for i, airline in enumerate(airlines[:3]):
            price = round(random.uniform(300, 1200) * passengers, 2)
            flights.append({
                "flight_id": f"FL{100 + i}",
                "airline": airline,
                "origin": origin,
                "destination": destination,
                "date": date,
                "duration_hours": round(random.uniform(2, 14), 1),
                "price_eur": price,
                "seats_available": random.randint(1, 50),
            })

        return {
            "status": "success",
            "flights": flights,
            "total_found": len(flights),
        }
    except Exception as e:
        return {"error": f"Flight search failed: {str(e)}"}


def estimate_flight_price(origin: str, destination: str) -> dict:
    """
    Estime le prix d'un vol entre deux villes.
    Args:
        origin: Ville de départ.
        destination: Ville d'arrivée.
    Returns:
        Dict avec prix minimum et maximum estimés en EUR.
    """
    try:
        import random
        base = random.randint(150, 800)
        return {
            "status": "success",
            "origin": origin,
            "destination": destination,
            "estimated_min_eur": base,
            "estimated_max_eur": base + random.randint(100, 600),
        }
    except Exception as e:
        return {"error": f"Estimation échouée: {str(e)}"}


# ─────────────────────────────────────────────
# HOTELS TOOLS
# ─────────────────────────────────────────────

def search_hotels(
    city: str,
    check_in: str,
    check_out: str,
    guests: int = 1,
    max_price_per_night: Optional[float] = None
) -> dict:
    """
    Search for available hotels in a city for given dates.

    Args:
        city: Destination city name.
        check_in: Check-in date in YYYY-MM-DD format.
        check_out: Check-out date in YYYY-MM-DD format.
        guests: Number of guests (default 1).
        max_price_per_night: Optional maximum price per night in EUR.

    Returns:
        A dict with a list of available hotels and their details.
    """
    try:
        if not city or not check_in or not check_out:
            return {"error": "city, check_in and check_out are required."}

        hotel_names = [
            f"{city} Grand Hotel",
            f"Hotel du Centre {city}",
            f"{city} Budget Inn",
            f"The {city} Suite",
        ]
        hotels = []
        for i, name in enumerate(hotel_names):
            price = round(random.uniform(60, 400), 2)
            if max_price_per_night and price > max_price_per_night:
                continue
            hotels.append({
                "hotel_id": f"H{200 + i}",
                "name": name,
                "city": city,
                "stars": random.randint(2, 5),
                "price_per_night_eur": price,
                "check_in": check_in,
                "check_out": check_out,
                "available_rooms": random.randint(1, 20),
                "rating": round(random.uniform(6.5, 9.8), 1),
            })

        return {
            "status": "success",
            "hotels": hotels,
            "total_found": len(hotels),
        }
    except Exception as e:
        return {"error": f"Hotel search failed: {str(e)}"}


# ─────────────────────────────────────────────
# ACTIVITIES TOOLS
# ─────────────────────────────────────────────

def search_activities(
    city: str,
    category: Optional[str] = None
) -> dict:
    """
    Search for tourist activities and things to do in a city.

    Args:
        city: City to search activities in.
        category: Optional category filter (e.g. "museum", "outdoor", "food").

    Returns:
        A dict with a list of recommended activities.
    """
    try:
        if not city:
            return {"error": "city is required."}

        all_activities = [
            {"name": f"Visit {city} Old Town", "category": "sightseeing", "price_eur": 0, "duration_hours": 2},
            {"name": f"{city} Food Tour", "category": "food", "price_eur": 45, "duration_hours": 3},
            {"name": f"{city} Museum of History", "category": "museum", "price_eur": 15, "duration_hours": 2},
            {"name": f"Hiking around {city}", "category": "outdoor", "price_eur": 25, "duration_hours": 5},
            {"name": f"{city} Cooking Class", "category": "food", "price_eur": 80, "duration_hours": 3},
        ]

        if category:
            activities = [a for a in all_activities if a["category"] == category.lower()]
        else:
            activities = all_activities

        return {
            "status": "success",
            "city": city,
            "activities": activities,
            "total_found": len(activities),
        }
    except Exception as e:
        return {"error": f"Activity search failed: {str(e)}"}


# ─────────────────────────────────────────────
# BUDGET TOOL
# ─────────────────────────────────────────────

def calculate_budget(
    flight_price: float,
    hotel_price_per_night: float,
    num_nights: int,
    activities_budget: float = 0.0,
    daily_food_budget: float = 50.0
) -> dict:
    """
    Calculate the total estimated budget for a trip.

    Args:
        flight_price: Total flight price in EUR.
        hotel_price_per_night: Hotel price per night in EUR.
        num_nights: Number of nights of the stay.
        activities_budget: Budget for activities in EUR (default 0).
        daily_food_budget: Daily food budget in EUR per person (default 50).

    Returns:
        A dict with a detailed budget breakdown and total.
    """
    try:
        hotel_total = hotel_price_per_night * num_nights
        food_total = daily_food_budget * num_nights
        total = flight_price + hotel_total + activities_budget + food_total

        return {
            "status": "success",
            "breakdown": {
                "flights_eur": round(flight_price, 2),
                "hotel_eur": round(hotel_total, 2),
                "activities_eur": round(activities_budget, 2),
                "food_eur": round(food_total, 2),
            },
            "total_eur": round(total, 2),
            "num_nights": num_nights,
        }
    except Exception as e:
        return {"error": f"Budget calculation failed: {str(e)}"}


# ─────────────────────────────────────────────
# WEATHER TOOL
# ─────────────────────────────────────────────

def get_weather_forecast(
    city: str,
    date: str
) -> dict:
    """
    Get a simulated weather forecast for a city on a specific date.

    Args:
        city: City name.
        date: Date in YYYY-MM-DD format.

    Returns:
        A dict with weather forecast details.
    """
    try:
        if not city or not date:
            return {"error": "city and date are required."}

        conditions = ["Sunny", "Partly Cloudy", "Cloudy", "Rainy", "Windy"]
        return {
            "status": "success",
            "city": city,
            "date": date,
            "condition": random.choice(conditions),
            "temperature_celsius": random.randint(10, 35),
            "humidity_percent": random.randint(30, 90),
            "wind_kmh": random.randint(5, 50),
        }
    except Exception as e:
        return {"error": f"Weather forecast failed: {str(e)}"}