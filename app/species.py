"""Curated fish taxonomy used by the zero-shot species classifier.

BioCLIP-2 predicts over this closed list.  The list intentionally contains
common, visually distinctive species that a person can present to a webcam.
Adding a species is as simple as appending an entry; the text embeddings are
rebuilt at startup.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Species:
    name: str
    scientific: str

    @property
    def prompts(self) -> tuple[str, str, str, str]:
        return (
            f"a photo of a {self.name}.",
            f"a photo of {self.scientific}.",
            f"a close-up photo of a {self.name}.",
            f"a photo of a fish, {self.name} ({self.scientific}).",
        )


SPECIES: tuple[Species, ...] = (
    Species("Clownfish", "Amphiprion ocellaris"),
    Species("Tilapia", "Oreochromis niloticus"),
    Species("Goldfish", "Carassius auratus"),
    Species("Koi", "Cyprinus rubrofuscus"),
    Species("Common carp", "Cyprinus carpio"),
    Species("Rainbow trout", "Oncorhynchus mykiss"),
    Species("Brown trout", "Salmo trutta"),
    Species("Atlantic salmon", "Salmo salar"),
    Species("Arctic char", "Salvelinus alpinus"),
    Species("Sea bass", "Dicentrarchus labrax"),
    Species("Catfish", "Ictalurus punctatus"),
    Species("Pufferfish", "Tetraodon mbu"),
    Species("Great white shark", "Carcharodon carcharias"),
    Species("Tiger shark", "Galeocerdo cuvier"),
    Species("Hammerhead shark", "Sphyrna mokarran"),
    Species("Whale shark", "Rhincodon typus"),
    Species("Lionfish", "Pterois volitans"),
    Species("Scorpionfish", "Scorpaena scrofa"),
    Species("Blue tang", "Paracanthurus hepatus"),
    Species("Angelfish", "Pomacanthus imperator"),
    Species("Butterflyfish", "Chaetodon"),
    Species("Parrotfish", "Scarus"),
    Species("Betta", "Betta splendens"),
    Species("Guppy", "Poecilia reticulata"),
    Species("Molly", "Poecilia sphenops"),
    Species("Neon tetra", "Paracheirodon innesi"),
    Species("Discus", "Symphysodon aequifasciatus"),
    Species("Oscar", "Astronotus ocellatus"),
    Species("Gourami", "Trichopodus pectoralis"),
    Species("Arowana", "Osteoglossum bicirrhosum"),
    Species("Seahorse", "Hippocampus kuda"),
    Species("Cod", "Gadus morhua"),
    Species("Haddock", "Melanogrammus aeglefinus"),
    Species("Hake", "Merluccius merluccius"),
    Species("Pollock", "Pollachius pollachius"),
    Species("Halibut", "Hippoglossus hippoglossus"),
    Species("Flounder", "Platichthys flesus"),
    Species("Sole", "Solea solea"),
    Species("Turbot", "Scophthalmus maximus"),
    Species("Mackerel", "Scomber scombrus"),
    Species("Tuna", "Thunnus thynnus"),
    Species("Sardine", "Sardina pilchardus"),
    Species("Herring", "Clupea harengus"),
    Species("Anchovy", "Engraulis encrasicolus"),
    Species("Barracuda", "Sphyraena barracuda"),
    Species("Swordfish", "Xiphias gladius"),
    Species("Marlin", "Makaira nigricans"),
    Species("Mahi-mahi", "Coryphaena hippurus"),
    Species("Snapper", "Lutjanus campechanus"),
    Species("Grouper", "Epinephelus marginatus"),
    Species("Mullet", "Mugil cephalus"),
    Species("Bream", "Abramis brama"),
    Species("Perch", "Perca fluviatilis"),
    Species("Pike", "Esox lucius"),
    Species("Zander", "Sander lucioperca"),
    Species("Eel", "Anguilla anguilla"),
    Species("Sturgeon", "Acipenser sturio"),
    Species("Milkfish", "Chanos chanos"),
    Species("Stingray", "Dasyatis pastinaca"),
    Species("Manta ray", "Mobula birostris"),
    Species("Piranha", "Pygocentrus nattereri"),
    Species("Peacock bass", "Cichla ocellaris"),
    Species("Arapaima", "Arapaima gigas"),
    Species("Ocean sunfish", "Mola mola"),
    Species("Largemouth bass", "Micropterus salmoides"),
    Species("Bluegill", "Lepomis macrochirus"),
    Species("Walleye", "Sander vitreus"),
    Species("Rohu", "Labeo rohita"),
    Species("Catla", "Labeo catla"),
    Species("Tench", "Tinca tinca"),
    Species("Roach", "Rutilus rutilus"),
    Species("Goby", "Gobius"),
    Species("Cichlid", "Cichlidae"),
    Species("Loach", "Misgurnus"),
    Species("Monkfish", "Lophius piscatorius"),
    Species("Tarpon", "Megalops atlanticus"),
    Species("Bonefish", "Albula vulpes"),
    Species("Wrasse", "Labridae"),
    Species("Damselfish", "Pomacentridae"),
    Species("Triggerfish", "Balistidae"),
)

SPECIES_NAMES: tuple[str, ...] = tuple(s.name for s in SPECIES)
SPECIES_BY_NAME: dict[str, Species] = {s.name: s for s in SPECIES}
SPECIES_BY_LOWER: dict[str, Species] = {s.name.lower(): s for s in SPECIES}

# ImageNet classes that are true fish (or fish-like enough that a high
# aggregate probability is useful evidence).  Deliberately excludes mammals,
# starfish, jellyfish and crayfish -- they must be "Not Fish".
IMAGENET_FISH_CLASSES: frozenset[str] = frozenset(
    {
        "tench",
        "goldfish",
        "great white shark",
        "tiger shark",
        "hammerhead",
        "electric ray",
        "stingray",
        "barracouta",
        "eel",
        "coho",
        "rock beauty",
        "anemone fish",
        "sturgeon",
        "gar",
        "lionfish",
        "puffer",
    }
)

# Used only to break a near-tie between two BioCLIP candidates when the
# ImageNet verifier is confident.  Values are names in SPECIES.
IMAGENET_TO_SPECIES: dict[str, str] = {
    "anemone fish": "Clownfish",
    "goldfish": "Goldfish",
    "great white shark": "Great white shark",
    "tiger shark": "Tiger shark",
    "hammerhead": "Hammerhead shark",
    "stingray": "Stingray",
    "eel": "Eel",
    "lionfish": "Lionfish",
    "puffer": "Pufferfish",
    "tench": "Tench",
    "sturgeon": "Sturgeon",
    "rock beauty": "Angelfish",
}
