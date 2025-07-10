#!/usr/bin/env python3
"""
Tariff Configuration System
===========================

Unified backend for handling different types of tariff policy configurations:
- Uniform rates across all countries/sectors
- Custom rates by country pairs
- Custom rates by sectors
- Custom rates by country-sector combinations

This provides a single data structure and processing pipeline for all tariff scenarios.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Set, Optional
import json
from pathlib import Path


@dataclass
class TariffConfiguration:
    """
    Unified tariff configuration for all policy types.
    
    Core design: All tariff policies are represented as a mapping from
    (importer, exporter, sector) triplets to tariff rates (in percentage).
    """
    
    # Core data: maps (importer, exporter, sector) → tariff_rate (%)
    tariff_changes: Dict[Tuple[str, str, str], float] = field(default_factory=dict)
    
    # Metadata
    policy_type: str = "custom"
    description: str = "Custom tariff configuration"
    
    # Reference lists (for validation)
    countries: List[str] = field(default_factory=list)
    sectors: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        """Validate the configuration after initialization."""
        self.validate()
    
    def validate(self) -> bool:
        """
        Validate the tariff configuration.
        
        Returns
        -------
        bool
            True if valid
            
        Raises
        ------
        ValueError
            If configuration is invalid
        """
        # Check that all tariff rates are non-negative
        for (imp, exp, sec), rate in self.tariff_changes.items():
            if rate < 0:
                raise ValueError(f"Negative tariff rate: {rate}% for ({imp}, {exp}, {sec})")
        
        # Check that countries and sectors in tariff_changes are in reference lists
        if self.countries:
            tariff_countries = {imp for imp, exp, sec in self.tariff_changes.keys()} | \
                              {exp for imp, exp, sec in self.tariff_changes.keys()}
            unknown_countries = tariff_countries - set(self.countries)
            if unknown_countries:
                raise ValueError(f"Unknown countries in tariff configuration: {unknown_countries}")
        
        if self.sectors:
            tariff_sectors = {sec for imp, exp, sec in self.tariff_changes.keys()}
            unknown_sectors = tariff_sectors - set(self.sectors)
            if unknown_sectors:
                raise ValueError(f"Unknown sectors in tariff configuration: {unknown_sectors}")
        
        return True
    
    @property
    def affected_countries(self) -> Set[str]:
        """Get set of all countries affected by tariff changes."""
        countries = set()
        for imp, exp, sec in self.tariff_changes.keys():
            countries.add(imp)
            countries.add(exp)
        return countries
    
    @property
    def affected_sectors(self) -> Set[str]:
        """Get set of all sectors affected by tariff changes."""
        return {sec for imp, exp, sec in self.tariff_changes.keys()}
    
    @property
    def num_changes(self) -> int:
        """Get number of tariff changes."""
        return len(self.tariff_changes)
    
    def get_tariff_rate(self, importer: str, exporter: str, sector: str) -> float:
        """
        Get tariff rate for specific importer-exporter-sector combination.
        
        Parameters
        ----------
        importer, exporter, sector : str
            Trade flow specification
            
        Returns
        -------
        float
            Tariff rate in percentage (0.0 if no change specified)
        """
        return self.tariff_changes.get((importer, exporter, sector), 0.0)
    
    def set_tariff_rate(self, importer: str, exporter: str, sector: str, rate: float) -> None:
        """Set tariff rate for specific importer-exporter-sector combination."""
        if rate < 0:
            raise ValueError(f"Tariff rate must be non-negative, got {rate}")
        
        if rate == 0.0:
            # Remove zero rates to keep dictionary clean
            self.tariff_changes.pop((importer, exporter, sector), None)
        else:
            self.tariff_changes[(importer, exporter, sector)] = rate
    
    @classmethod
    def from_uniform_rate(cls, 
                         rate: float, 
                         countries: List[str], 
                         sectors: List[str],
                         description: str | None = None) -> "TariffConfiguration":
        """
        Create tariff configuration with uniform rate across all country-sector combinations.
        
        Parameters
        ----------
        rate : float
            Uniform tariff rate in percentage
        countries : List[str]
            List of country names
        sectors : List[str]
            List of sector names
        description : str, optional
            Description of the policy
            
        Returns
        -------
        TariffConfiguration
            Configured tariff policy
        """
        if rate < 0:
            raise ValueError(f"Tariff rate must be non-negative, got {rate}")
        
        tariff_changes = {}
        
        # Apply uniform rate to all importer-exporter-sector combinations
        # except domestic trade (importer == exporter)
        for importer in countries:
            for exporter in countries:
                if importer != exporter:  # No tariffs on domestic trade
                    for sector in sectors:
                        if rate > 0:  # Only store non-zero rates
                            tariff_changes[(importer, exporter, sector)] = rate
        
        desc = description or f"Uniform {rate}% tariff on all imports"
        
        return cls(
            tariff_changes=tariff_changes,
            policy_type="uniform",
            description=desc,
            countries=countries.copy(),
            sectors=sectors.copy()
        )
    
    @classmethod
    def from_country_rates(cls,
                          country_rates: Dict[Tuple[str, str], float],
                          sectors: List[str],
                          countries: List[str] | None = None,
                          description: str | None = None) -> "TariffConfiguration":
        """
        Create tariff configuration with rates specified by country pairs.
        
        Parameters
        ----------
        country_rates : Dict[Tuple[str, str], float]
            Mapping from (importer, exporter) pairs to tariff rates
        sectors : List[str]
            List of sector names (rate applies to all sectors)
        countries : List[str], optional
            Full list of countries for validation
        description : str, optional
            Description of the policy
            
        Returns
        -------
        TariffConfiguration
            Configured tariff policy
        """
        tariff_changes = {}
        
        # Apply country-specific rates to all sectors
        for (importer, exporter), rate in country_rates.items():
            if rate < 0:
                raise ValueError(f"Tariff rate must be non-negative, got {rate} for {importer}→{exporter}")
            
            if importer == exporter:
                raise ValueError(f"Cannot set tariff for domestic trade: {importer}→{exporter}")
            
            if rate > 0:  # Only store non-zero rates
                for sector in sectors:
                    tariff_changes[(importer, exporter, sector)] = rate
        
        # Infer countries from rates if not provided
        if countries is None:
            countries_set = set()
            for imp, exp in country_rates.keys():
                countries_set.add(imp)
                countries_set.add(exp)
            countries = sorted(list(countries_set))
        
        desc = description or f"Country-specific tariffs ({len(country_rates)} country pairs)"
        
        return cls(
            tariff_changes=tariff_changes,
            policy_type="by_country",
            description=desc,
            countries=countries.copy(),
            sectors=sectors.copy()
        )
    
    @classmethod
    def from_sector_rates(cls,
                         sector_rates: Dict[str, float],
                         countries: List[str],
                         description: str | None = None) -> "TariffConfiguration":
        """
        Create tariff configuration with rates specified by sectors.
        
        Parameters
        ----------
        sector_rates : Dict[str, float]
            Mapping from sector names to tariff rates
        countries : List[str]
            List of country names (rate applies to all country pairs)
        description : str, optional
            Description of the policy
            
        Returns
        -------
        TariffConfiguration
            Configured tariff policy
        """
        tariff_changes = {}
        
        # Apply sector-specific rates to all country pairs
        for sector, rate in sector_rates.items():
            if rate < 0:
                raise ValueError(f"Tariff rate must be non-negative, got {rate} for sector {sector}")
            
            if rate > 0:  # Only store non-zero rates
                for importer in countries:
                    for exporter in countries:
                        if importer != exporter:  # No tariffs on domestic trade
                            tariff_changes[(importer, exporter, sector)] = rate
        
        desc = description or f"Sector-specific tariffs ({len(sector_rates)} sectors)"
        
        return cls(
            tariff_changes=tariff_changes,
            policy_type="by_sector", 
            description=desc,
            countries=countries.copy(),
            sectors=list(sector_rates.keys())
        )
    
    @classmethod
    def from_country_sector_rates(cls,
                                 rates: Dict[Tuple[str, str, str], float],
                                 countries: List[str] | None = None,
                                 sectors: List[str] | None = None,
                                 description: str | None = None) -> "TariffConfiguration":
        """
        Create tariff configuration with rates specified by country-sector combinations.
        
        Parameters
        ----------
        rates : Dict[Tuple[str, str, str], float]
            Mapping from (importer, exporter, sector) triplets to tariff rates
        countries : List[str], optional
            Full list of countries for validation
        sectors : List[str], optional
            Full list of sectors for validation
        description : str, optional
            Description of the policy
            
        Returns
        -------
        TariffConfiguration
            Configured tariff policy
        """
        tariff_changes = {}
        
        # Validate and store rates
        for (importer, exporter, sector), rate in rates.items():
            if rate < 0:
                raise ValueError(f"Tariff rate must be non-negative, got {rate} for {importer}→{exporter}:{sector}")
            
            if importer == exporter:
                raise ValueError(f"Cannot set tariff for domestic trade: {importer}→{exporter}:{sector}")
            
            if rate > 0:  # Only store non-zero rates
                tariff_changes[(importer, exporter, sector)] = rate
        
        # Infer countries and sectors if not provided
        if countries is None:
            countries_set = set()
            for imp, exp, sec in rates.keys():
                countries_set.add(imp)
                countries_set.add(exp)
            countries = sorted(list(countries_set))
        
        if sectors is None:
            sectors = sorted(list({sec for imp, exp, sec in rates.keys()}))
        
        desc = description or f"Country-sector specific tariffs ({len(tariff_changes)} combinations)"
        
        return cls(
            tariff_changes=tariff_changes,
            policy_type="by_country_sector",
            description=desc,
            countries=countries.copy(),
            sectors=sectors.copy()
        )
    
    def save_to_json(self, filename: str) -> None:
        """Save configuration to JSON file."""
        data = {
            'tariff_changes': {f"{imp}|{exp}|{sec}": rate for (imp, exp, sec), rate in self.tariff_changes.items()},
            'policy_type': self.policy_type,
            'description': self.description,
            'countries': self.countries,
            'sectors': self.sectors
        }
        
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
    
    @classmethod
    def load_from_json(cls, filename: str) -> "TariffConfiguration":
        """Load configuration from JSON file."""
        with open(filename, 'r') as f:
            data = json.load(f)
        
        # Convert string keys back to tuples
        tariff_changes = {}
        for key_str, rate in data['tariff_changes'].items():
            imp, exp, sec = key_str.split('|')
            tariff_changes[(imp, exp, sec)] = rate
        
        return cls(
            tariff_changes=tariff_changes,
            policy_type=data.get('policy_type', 'custom'),
            description=data.get('description', 'Loaded from JSON'),
            countries=data.get('countries', []),
            sectors=data.get('sectors', [])
        )
    
    def summary(self) -> str:
        """Get a summary string of the tariff configuration."""
        lines = [
            f"Tariff Configuration: {self.description}",
            f"Policy Type: {self.policy_type}",
            f"Number of Changes: {self.num_changes}",
            f"Affected Countries: {len(self.affected_countries)} ({', '.join(sorted(self.affected_countries))})",
            f"Affected Sectors: {len(self.affected_sectors)} ({', '.join(sorted(self.affected_sectors))})"
        ]
        
        if self.num_changes <= 10:
            lines.append("Tariff Changes:")
            for (imp, exp, sec), rate in sorted(self.tariff_changes.items()):
                lines.append(f"  {imp} → {exp} ({sec}): {rate}%")
        
        return "\n".join(lines)


# Demo functions for testing different configurations
def demo_uniform_tariff():
    """Demo: Uniform 25% tariff on all imports."""
    countries = ["USA", "CHN", "DEU", "JPN"]
    sectors = ["Agriculture", "Manufacturing", "Services"]
    
    config = TariffConfiguration.from_uniform_rate(
        rate=25.0,
        countries=countries,
        sectors=sectors,
        description="25% uniform tariff on all imports"
    )
    
    print("=== UNIFORM TARIFF DEMO ===")
    print(config.summary())
    return config


def demo_country_tariffs():
    """Demo: Country-specific tariffs."""
    country_rates = {
        ("USA", "CHN"): 15.0,
        ("USA", "DEU"): 5.0,
        ("USA", "JPN"): 10.0
    }
    sectors = ["Agriculture", "Manufacturing", "Services"]
    
    config = TariffConfiguration.from_country_rates(
        country_rates=country_rates,
        sectors=sectors,
        description="US tariffs on selected countries"
    )
    
    print("\n=== COUNTRY-SPECIFIC TARIFF DEMO ===")
    print(config.summary())
    return config


def demo_sector_tariffs():
    """Demo: Sector-specific tariffs."""
    sector_rates = {
        "Agriculture": 10.0,
        "Manufacturing": 20.0,
        "Services": 5.0
    }
    countries = ["USA", "CHN", "DEU", "JPN"]
    
    config = TariffConfiguration.from_sector_rates(
        sector_rates=sector_rates,
        countries=countries,
        description="Sector-specific tariff policy"
    )
    
    print("\n=== SECTOR-SPECIFIC TARIFF DEMO ===")
    print(config.summary())
    return config


def demo_country_sector_tariffs():
    """Demo: Country-sector specific tariffs."""
    rates = {
        ("USA", "CHN", "Manufacturing"): 50.0,
        ("USA", "CHN", "Agriculture"): 15.0,
        ("USA", "DEU", "Services"): 8.0,
        ("JPN", "CHN", "Manufacturing"): 30.0
    }
    
    config = TariffConfiguration.from_country_sector_rates(
        rates=rates,
        description="Strategic country-sector tariffs"
    )
    
    print("\n=== COUNTRY-SECTOR SPECIFIC TARIFF DEMO ===")
    print(config.summary())
    return config


if __name__ == "__main__":
    print("Tariff Configuration System Demo")
    print("=" * 50)
    
    # Run all demos
    uniform_config = demo_uniform_tariff()
    country_config = demo_country_tariffs()
    sector_config = demo_sector_tariffs()
    country_sector_config = demo_country_sector_tariffs()
    
    # Demo file I/O
    print("\n=== FILE I/O DEMO ===")
    uniform_config.save_to_json("demo_uniform_tariffs.json")
    loaded_config = TariffConfiguration.load_from_json("demo_uniform_tariffs.json")
    print(f"Saved and loaded config matches: {uniform_config.tariff_changes == loaded_config.tariff_changes}")
    
    print("\n✅ All demos completed successfully!") 