"""Coastal places the website can assess, besides our three guided spots.

A "spot" has a guide and a trained history; a "place" is any point on the coast. Each
point here was checked to have wave data (a city centre on land gets none). Adding a place
is one line. (Stage 11 will move this list into the database.)
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Place:
    name: str
    region: str  # province
    latitude: float
    longitude: float


PLACES = [
    Place("Mezitli sahili", "Mersin", 36.735, 34.530),
    Place("Karataş", "Adana", 36.555, 35.385),
    Place("İskenderun", "Hatay", 36.600, 36.170),
    Place("Konyaaltı", "Antalya", 36.855, 30.620),
    Place("Kemer", "Antalya", 36.600, 30.570),
    Place("Side", "Antalya", 36.765, 31.385),
    Place("Alanya", "Antalya", 36.535, 31.990),
    Place("Kaş", "Antalya", 36.195, 29.640),
    Place("Fethiye Çalış", "Muğla", 36.665, 29.095),
    Place("Marmaris", "Muğla", 36.840, 28.270),
    Place("Datça", "Muğla", 36.725, 27.685),
    Place("Bodrum Bitez", "Muğla", 37.025, 27.390),
    Place("Didim Altınkum", "Aydın", 37.350, 27.260),
    Place("Seferihisar Sığacık", "İzmir", 38.195, 26.785),
    Place("Urla Gülbahçe", "İzmir", 38.355, 26.640),
    Place("Foça", "İzmir", 38.665, 26.745),
    Place("Ayvalık Sarımsaklı", "Balıkesir", 39.245, 26.665),
    Place("Assos", "Çanakkale", 39.485, 26.335),
    Place("Bozcaada", "Çanakkale", 39.825, 26.040),
    Place("Gökçeada Aydıncık", "Çanakkale", 40.130, 25.965),
    Place("Kilyos", "İstanbul", 41.255, 29.030),
    Place("Şile", "İstanbul", 41.180, 29.610),
    Place("Akçakoca", "Düzce", 41.090, 31.120),
    Place("Sinop", "Sinop", 42.020, 35.160),
    Place("Atakum", "Samsun", 41.335, 36.265),
    Place("Trabzon", "Trabzon", 41.010, 39.720),
]
