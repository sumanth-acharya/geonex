"""
Fix 2: Replace DETAILED_REPORTS with ASCII-safe structured technical reports
using Python's repr() to ensure clean string literals.
"""
import re

with open('app.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Build new DETAILED_REPORTS as a proper Python dict literal
TECH = {}

TECH["airplane"] = (
    "[SPECTRAL SIGNATURE]\n"
    "- VIS reflectance: High on metallic fuselage surfaces\n"
    "- NIR reflectance: High (aluminium/composite specular)\n"
    "- Thermal: Low emissivity (metallic surface)\n"
    "- Shadow: Elongated bilateral -- diagnostic of fixed-wing geometry\n\n"
    "[SPATIAL METRICS]\n"
    "- Object length: 20-80m | Wingspan: 15-65m (GSD-normalised)\n"
    "- Aspect ratio: ~1.3:1 (fuselage:wingspan)\n"
    "- Texture: Smooth, specular reflection on tarmac\n"
    "- Shape: Bilateral symmetry -- strong discriminator vs ground vehicles\n\n"
    "[CLASSIFICATION FINDINGS]\n"
    "- Class: Airplane (object-level detection, airport/airfield context)\n"
    "- Model: MobileNetV2 softmax probability output\n"
    "- Training similarity: NWPU-RESISC45 airplane tiles (GSD 0.2-30m)\n"
    "- Contextual cues: Tarmac, taxiway markings, apron proximity\n\n"
    "[ENVIRONMENTAL INDICATORS]\n"
    "- LST: Elevated (reflective tarmac + engine heat)\n"
    "- ISC: >95% in surrounding apron zone\n"
    "- NDVI: ~0.0 (no vegetation)\n\n"
    "[APPLICATIONS]\n"
    "SAR/optical fusion for aircraft detection | Air-traffic density monitoring | "
    "Airport capacity planning | Defense intelligence | Emergency airlift coordination"
)

TECH["airport"] = (
    "[SPECTRAL SIGNATURE]\n"
    "- Runway surface: Grey asphalt, moderate VIS reflectance (R~0.15, G~0.14, B~0.13)\n"
    "- Terminal rooftops: High NIR reflectance (metal/concrete)\n"
    "- NDVI: ~0.0-0.05 (dominantly impervious surface)\n"
    "- Thermal: Elevated LST across runway and apron surfaces\n\n"
    "[SPATIAL METRICS]\n"
    "- Runway: 1,800-4,500m length x 30-60m width\n"
    "- Terminal footprint: 10,000-200,000 m2\n"
    "- Runway orientation: Aligned with prevailing wind (ILS/VOR indicator)\n"
    "- Texture: Fine on runways (painted markings), coarse on surrounding zones\n\n"
    "[CLASSIFICATION FINDINGS]\n"
    "- Class: Airport (facility-level land-use classification)\n"
    "- Key discriminators: Parallel runway geometry, apron-terminal adjacency\n"
    "- Training match: High spatial frequency linear features + low NDVI\n"
    "- False positive risk: Industrial zones (similar ISC) -- differentiated by runway geometry\n\n"
    "[ENVIRONMENTAL INDICATORS]\n"
    "- ISC: >80% (major impervious zone)\n"
    "- Emissions: NOx/VOC point sources (combustion)\n"
    "- Stormwater risk: High (glycol de-icing, jet fuel runoff)\n"
    "- UHI contribution: Significant\n\n"
    "[APPLICATIONS]\n"
    "Airport expansion monitoring | Runway capacity analysis | Noise contour mapping | "
    "Environmental compliance (ICAO Annex 16) | Emergency runway assessment"
)

TECH["forest"] = (
    "[SPECTRAL SIGNATURE]\n"
    "- NDVI: 0.65-0.90 (dense, healthy broadleaf/mixed canopy)\n"
    "- NIR reflectance: Very high (0.45-0.55) -- cell structure scattering\n"
    "- Red band: Strong absorption (0.04-0.08) -- chlorophyll a/b\n"
    "- EVI (Enhanced Vegetation Index): 0.4-0.7\n"
    "- NDWI (Gao): 0.1-0.4 (canopy water content indicator)\n"
    "- SWIR: Moderate -- canopy moisture stress indicator\n\n"
    "[SPATIAL METRICS]\n"
    "- Canopy closure: >75%\n"
    "- Texture: High spatial variance (crown shadow heterogeneity)\n"
    "- Fractal dimension: 1.6-1.9 (complex edge geometry)\n"
    "- Crown diameter (GSD-dependent): 3-25m detectable\n\n"
    "[CLASSIFICATION FINDINGS]\n"
    "- Class: Forest -- Dense Canopy Land Cover (LCCS Class A1.1)\n"
    "- Training similarity: NWPU-RESISC45 forest tiles (strong spectral-spatial match)\n"
    "- Confusion risk: Chaparral (lower NDVI) | Meadow (no tree shadow texture)\n"
    "- CNN features extracted: Edge gradients, crown texture, shadow pattern\n\n"
    "[ENVIRONMENTAL INDICATORS]\n"
    "- Carbon Stock: 80-300 tC/ha (biome-dependent)\n"
    "- Above-Ground Biomass (AGB): Very High (SAR/LiDAR for precise estimate)\n"
    "- LST suppression: 3-8 deg C cooler than surrounding non-vegetated land\n"
    "- Evapotranspiration: High (latent heat flux dominant)\n"
    "- Fire Risk Index: Moderate (moisture-dependent)\n\n"
    "[APPLICATIONS]\n"
    "REDD+ carbon credit verification | AGB mapping | Deforestation alert (GLAD/PRODES) | "
    "Wildfire fuel load assessment | Biodiversity hotspot identification | Watershed zoning"
)

TECH["beach"] = (
    "[SPECTRAL SIGNATURE]\n"
    "- Sandy substrate: Very high VIS reflectance (R~0.55, G~0.50, B~0.45)\n"
    "- Shoreline boundary: Sharp spectral contrast (sand ~0.5 vs water ~0.02-0.05)\n"
    "- NDWI (McFeeters): Positive in intertidal zone; negative on dry sand\n"
    "- NIR: Low (non-vegetated) -- distinguishes from dune grasslands\n\n"
    "[SPATIAL METRICS]\n"
    "- Shoreline sinuosity index: 1.0-1.3 (linear to mildly curved)\n"
    "- Beach width (GSD-relative): 5-300m\n"
    "- Texture: Low spatial frequency (smooth sand surface)\n"
    "- Wet/dry sand boundary: Detectable at <5m GSD\n\n"
    "[CLASSIFICATION FINDINGS]\n"
    "- Class: Beach (coastal sedimentary depositional environment)\n"
    "- Key discriminators: High-albedo strip adjacent to dark water body\n"
    "- Confusion risk: Dry riverbeds (similar spectral) -- differentiated by water adjacency\n"
    "- Training match: Strong NWPU-RESISC45 beach class alignment\n\n"
    "[ENVIRONMENTAL INDICATORS]\n"
    "- Shoreline Change Rate: Detectable via multi-temporal NDWI/DSAS\n"
    "- Coastal Vulnerability Index (CVI): Moderate-High\n"
    "- Sea Level Rise sensitivity: High (low-elevation depositional zone)\n"
    "- Nesting habitat: Cheloniidae, Charadriiformes (protected)\n\n"
    "[APPLICATIONS]\n"
    "Shoreline change detection (DSAS) | Coastal erosion risk modelling | "
    "MPA delineation | Storm surge impact | Tourism carrying capacity"
)

TECH["river"] = (
    "[SPECTRAL SIGNATURE]\n"
    "- Water reflectance: Very low VIS (R~0.02-0.06) -- strong absorption\n"
    "- NIR: Near-zero (strong absorption) -- primary water discriminator\n"
    "- NDWI (McFeeters): Positive (>0.2) -- reliable water detection index\n"
    "- Turbidity: Elevated red/green ratio in sediment-laden water (TSS proxy)\n"
    "- Algal bloom: Green band anomaly (Chl-a indicator in eutrophic reaches)\n\n"
    "[SPATIAL METRICS]\n"
    "- Channel width: 5m-5km (scale-dependent)\n"
    "- Sinuosity index: 1.0 (straight channel) to >2.5 (meandering/anastomosing)\n"
    "- Braiding index: Multi-thread pattern indicates high bedload transport\n"
    "- Floodplain width: Detectable via NDWI expansion during flood events\n\n"
    "[CLASSIFICATION FINDINGS]\n"
    "- Class: River (Linear inland water body -- LCCS W1)\n"
    "- Key discriminators: Dark linear feature + riparian vegetation fringe + NDWI positive\n"
    "- Geomorphic signature: Sinuosity encodes discharge and sediment regime\n"
    "- Confusion risk: Road/shadow (similar geometry) -- differentiated by NDWI\n\n"
    "[ENVIRONMENTAL INDICATORS]\n"
    "- TSS (Suspended Sediment): Red/NIR band ratio proxy\n"
    "- Flood stage: NDWI expansion vs. baseline water mask\n"
    "- Riparian corridor NDVI: Target >0.4 for healthy buffer\n"
    "- Water quality anomalies: Algal bloom index, turbidity mapping\n\n"
    "[APPLICATIONS]\n"
    "Flood inundation mapping | Water quality monitoring (TSS, Chl-a) | "
    "Hydropower siting | Drought streamflow estimation | Aquatic habitat connectivity"
)

TECH["desert"] = (
    "[SPECTRAL SIGNATURE]\n"
    "- Bare soil: High VIS reflectance (R~0.35-0.60, spectrally flat to red-shifted)\n"
    "- NDVI: <0.1 (near-zero or negative -- bare mineral surface)\n"
    "- Albedo: 0.25-0.45 (sand) to 0.10-0.20 (dark reg/volcanic)\n"
    "- Mineralogy: Iron oxide -> red/orange hue | Quartz -> bright white\n"
    "- LST daytime: Very high (50-70 deg C surface temperature)\n\n"
    "[SPATIAL METRICS]\n"
    "- Surface roughness: Low (smooth ergs) to high (rocky reg/yardangs)\n"
    "- Dune wavelength: 50m-2km (detectable at moderate GSD)\n"
    "- Texture: Directional (wind-formed dune NE orientation dominant)\n"
    "- Sand encroachment front: Advancing bright spectral boundary\n\n"
    "[CLASSIFICATION FINDINGS]\n"
    "- Class: Desert/Hyper-arid Zone (Koppen BWh/BWk)\n"
    "- Key discriminators: High albedo + near-zero NDVI + no structural features\n"
    "- Sub-types: Erg (dunes) | Reg (gravel plain) | Hammada (rocky plateau)\n"
    "- Confusion risk: Dry riverbeds, bare agricultural soil -- context differentiates\n\n"
    "[ENVIRONMENTAL INDICATORS]\n"
    "- Rainfall: <250mm/yr (hyper-arid: <25mm/yr)\n"
    "- Desertification rate: Monitorable via NDVI trend and albedo change analysis\n"
    "- Dust emission potential: High (Aeolian PM10/PM2.5 source)\n"
    "- LST diurnal range: 30-50 deg C (low thermal inertia)\n\n"
    "[APPLICATIONS]\n"
    "Desertification monitoring (UNCCD) | CSP/PV site selection | "
    "Mineral exploration | Dune migration modelling | Dust storm source attribution"
)

TECH["lake"] = (
    "[SPECTRAL SIGNATURE]\n"
    "- Open water reflectance: Very low NIR (<0.05) -- strong absorption\n"
    "- NDWI: >0.3 (clean freshwater); reduced in turbid/eutrophic conditions\n"
    "- Turbidity index: Elevated red/green ratio (TSS proxy)\n"
    "- Cyanobacterial bloom: NIR shoulder at 700nm (MCI index detectable)\n"
    "- Depth indicator: Blue/green ratio (clearer water = higher blue fraction)\n\n"
    "[SPATIAL METRICS]\n"
    "- Shape index: Circular (volcanic) to elongate (glacial/tectonic origin)\n"
    "- Shoreline sinuosity: 1.0-2.5\n"
    "- Surface area: Direct measurement via NDWI thresholding\n"
    "- Fetch length: Wind-wave energy estimation key variable\n\n"
    "[CLASSIFICATION FINDINGS]\n"
    "- Class: Inland Water Body -- Lake (LCCS W1.1)\n"
    "- Key discriminators: Enclosed dark water mass with defined land boundary\n"
    "- Sub-types: Freshwater | Saline (elevated blue/green) | Seasonal (NDWI time-series)\n"
    "- Confusion risk: Large ponds, reservoirs -- size threshold differentiation\n\n"
    "[ENVIRONMENTAL INDICATORS]\n"
    "- Trophic state: Oligotrophic to Eutrophic (NDVI/Chl-a remote sensing proxies)\n"
    "- Storage volume change: Surface area x altimetry-derived bathymetry\n"
    "- Drought sensitivity: High -- area loss detectable in multi-temporal analysis\n"
    "- Methane source: Anaerobic sediment degassing (eutrophic lakes)\n\n"
    "[APPLICATIONS]\n"
    "Water quality monitoring (WHO compliance) | Reservoir storage estimation | "
    "Algal bloom early warning | Drought assessment | Ramsar wetland status"
)

TECH["mountain"] = (
    "[SPECTRAL SIGNATURE]\n"
    "- Rock surfaces: Variable reflectance (0.10-0.45) -- lithology dependent\n"
    "- Snow/ice cap: Very high albedo (0.60-0.90) -- dominates upper elevations\n"
    "- Vegetation belts: NDVI gradient 0.6-0.8 (montane) -> 0.2-0.4 (alpine) -> ~0 (barren)\n"
    "- Shadow fraction: High (steep illumination angle at high relief)\n"
    "- Thermal: Low LST at altitude (lapse rate ~6.5 deg C/1000m)\n\n"
    "[SPATIAL METRICS]\n"
    "- Relief amplitude: Hundreds to thousands of metres\n"
    "- Slope gradient: Inferred from shadow azimuth and DEM derivatives\n"
    "- Texture: Very high spatial variance (heterogeneous surface composition)\n"
    "- Ridge geometry: High fractal dimension (crenulated boundary pattern)\n\n"
    "[CLASSIFICATION FINDINGS]\n"
    "- Class: Mountain/Highland Terrain (Complex terrain -- LCCS)\n"
    "- Key discriminators: Shadow gradient + snow cap + vegetation zonation\n"
    "- Altitudinal zones: Montane / Subalpine / Alpine / Nival\n"
    "- Confusion risk: Chaparral at foothills -- differentiated by shadow intensity\n\n"
    "[ENVIRONMENTAL INDICATORS]\n"
    "- Snow Cover Area (SCA): Detectable via NDSI index\n"
    "- Glacier mass balance: Terminus retreat measurable via multi-temporal imagery\n"
    "- Landslide susceptibility: High on slopes >30 degrees\n"
    "- Permafrost: Detectable via thermal anomaly mapping\n\n"
    "[APPLICATIONS]\n"
    "Glacier mass balance (GRACE/ICESat) | Snowmelt runoff modelling | "
    "Landslide hazard mapping | Alpine biodiversity surveys | Avalanche risk zoning"
)

TECH["wetland"] = (
    "[SPECTRAL SIGNATURE]\n"
    "- Saturated soil: Low NIR, moderate VIS -- darker than dry soil\n"
    "- Emergent vegetation: Moderate NDVI (0.3-0.6) -- distinct from upland forest\n"
    "- Open water patches: NDWI positive (>0.2)\n"
    "- Peat/organic soil: Very low reflectance across all bands\n"
    "- Spectral pattern: Heterogeneous mosaic (NDVI/NDWI hybrid) -- diagnostic\n\n"
    "[SPATIAL METRICS]\n"
    "- Surface heterogeneity: High (interspersed water, emergent veg, mud)\n"
    "- Boundary geometry: Diffuse/gradational (not sharp)\n"
    "- Texture: Mixed fine-coarse by vegetation density\n"
    "- Seasonal extent: NDWI time-series essential for full mapping\n\n"
    "[CLASSIFICATION FINDINGS]\n"
    "- Class: Wetland (LCCS W2 -- Inland non-permanent/permanent water)\n"
    "- Sub-types: Marsh | Swamp | Mangrove (coastal) | Peatland | Floodplain\n"
    "- Key discriminators: Hybrid NDVI/NDWI pattern + seasonal inundation signal\n"
    "- Confusion risk: Rice paddy (similar signal) -- SAR-C backscatter differentiates\n\n"
    "[ENVIRONMENTAL INDICATORS]\n"
    "- Blue Carbon: 3-5x forest carbon per unit area (mangroves, peatlands)\n"
    "- Biodiversity: >35% of all species depend on wetland habitats\n"
    "- Flood regulation: Peak flow attenuation capacity\n"
    "- CH4 emissions: Significant anaerobic decomposition source\n\n"
    "[APPLICATIONS]\n"
    "Ramsar site monitoring | Blue carbon accounting | Flood storage mapping | "
    "Biodiversity conservation planning | Water purification service valuation"
)

TECH["dense_residential"] = (
    "[SPECTRAL SIGNATURE]\n"
    "- Rooftop mix: Red tile (high R, low B) | Asphalt (low VIS) | Concrete (flat moderate VIS)\n"
    "- NDVI: 0.0-0.15 (minimal vegetation between structures)\n"
    "- ISC: >75% (high impervious surface cover)\n"
    "- LST: High UHI hotspot (4-8 deg C above suburban fringe)\n"
    "- Shadow fraction: High (multi-storey structures, narrow lanes)\n\n"
    "[SPATIAL METRICS]\n"
    "- Building density: >60 units/ha\n"
    "- Plot coverage ratio: 0.7-0.9\n"
    "- Road lane width: 3-8m (narrow lanes)\n"
    "- Texture: Fine-grain, high spatial frequency\n\n"
    "[CLASSIFICATION FINDINGS]\n"
    "- Class: Dense Residential (Urban Built-up B1.1 -- LCCS)\n"
    "- Key discriminators: High rooftop density + fine texture + low NDVI\n"
    "- Differentiated from commercial: Smaller footprint size, denser packing pattern\n"
    "- Morphological type: Organic grid (informal) vs. regular grid (planned)\n\n"
    "[ENVIRONMENTAL INDICATORS]\n"
    "- UHI Intensity: High (>4 deg C above rural reference)\n"
    "- Green Space Ratio: <10% (below WHO 9m2/person recommendation)\n"
    "- Population density proxy: High (>10,000 persons/km2)\n"
    "- Flood runoff coefficient: 0.7-0.9 (high imperviousness)\n\n"
    "[APPLICATIONS]\n"
    "Population density modelling | Slum/informal settlement mapping | "
    "UHI mitigation planning | Infrastructure demand analysis | Census augmentation"
)

TECH["commercial_area"] = (
    "[SPECTRAL SIGNATURE]\n"
    "- Flat rooftops: Bright NIR (metal/EPDM) or moderate VIS (concrete/gravel)\n"
    "- NDVI: 0.0-0.1 (near-zero vegetation)\n"
    "- ISC: >85% (near-complete imperviousness)\n"
    "- LST: Very high daytime (reflective/absorptive flat roofs + HVAC waste heat)\n"
    "- Parking surfaces: Dark asphalt (low VIS reflectance 0.05-0.10)\n\n"
    "[SPATIAL METRICS]\n"
    "- Building footprint: Large (1,000-50,000 m2)\n"
    "- Roof-to-ground ratio: 0.5-0.8\n"
    "- Texture: Coarse within buildings, fine in parking zones\n"
    "- Road network: Wide arterials (8-30m) with loading bays\n\n"
    "[CLASSIFICATION FINDINGS]\n"
    "- Class: Commercial/Retail Zone (Urban Built-up B2 -- LCCS)\n"
    "- Key discriminators: Large homogeneous rooftops + expansive parking + road access\n"
    "- False positive: Industrial (similar ISC) -- differentiated by loading dock geometry\n\n"
    "[ENVIRONMENTAL INDICATORS]\n"
    "- LST: 5-10 deg C above vegetated land (peak afternoon)\n"
    "- Energy consumption: High per m2 (HVAC, lighting -- CO2 equivalent high)\n"
    "- Stormwater: Near-total impervious runoff (>0.85 runoff coefficient)\n\n"
    "[APPLICATIONS]\n"
    "Retail trade area mapping | UHI hotspot analysis | EV charging infrastructure siting | "
    "Commercial property assessment | Transport demand modelling"
)

TECH["industrial_area"] = (
    "[SPECTRAL SIGNATURE]\n"
    "- Metal roofs: Very high NIR reflectance (specular) + moderate VIS\n"
    "- Stockpile materials: Spectral mix of coal (black), limestone (white), ore (brown)\n"
    "- NDVI: ~0.0 (no vegetation in active industrial zone)\n"
    "- Thermal: Elevated point sources (kiln, furnace, cooling towers visible in TIR)\n"
    "- Smoke plumes: Detectable in TIR/SWIR bands\n\n"
    "[SPATIAL METRICS]\n"
    "- Building footprint: Very large (5,000-200,000 m2), low pitch roofs\n"
    "- ISC: >80%\n"
    "- Road access: Wide (HGV routing) with rail siding in heavy industry\n"
    "- Chimney shadows: Point source indicators (height from shadow length)\n\n"
    "[CLASSIFICATION FINDINGS]\n"
    "- Class: Industrial/Manufacturing Zone (B3 -- LCCS)\n"
    "- Key discriminators: Large-footprint buildings + material stockpiles + rail/road access\n"
    "- Pollution pressure: Chimney plume direction indicates prevailing wind vector\n\n"
    "[ENVIRONMENTAL INDICATORS]\n"
    "- Air pollution (PM2.5, SO2, NOx): detectable via Sentinel-5P TROPOMI\n"
    "- Groundwater contamination risk: High (heavy metals, solvents, leachate)\n"
    "- LST anomaly: Point source thermal signature visible in TIRS band\n\n"
    "[APPLICATIONS]\n"
    "Industrial emission attribution | Environmental compliance monitoring | "
    "Supply chain infrastructure mapping | Land contamination risk screening"
)

# General template for remaining classes
def general_tech(cls):
    name = cls.replace('_', ' ').title()
    return (
        "[SPECTRAL SIGNATURE]\n"
        "- VIS/NIR reflectance: Class-specific spectral profile matched to MobileNetV2 features\n"
        "- NDVI: Dependent on vegetation fraction within classified extent\n"
        "- LST: Varies by surface material and land-use activity intensity\n\n"
        "[SPATIAL METRICS]\n"
        "- Texture: Characteristic spatial frequency and pattern for " + name + "\n"
        "- Geometry: Distinctive shape descriptors extracted by CNN feature layers\n"
        "- Object scale: Consistent with training patch GSD (0.2-30m ground resolution)\n\n"
        "[CLASSIFICATION FINDINGS]\n"
        "- Class: " + name + " (LCCS-compliant land cover category)\n"
        "- Model: MobileNetV2 softmax probability output (transfer-learned on NWPU-RESISC45)\n"
        "- Training similarity: Spectral-spatial feature match to " + name + " training tiles\n"
        "- CNN feature layers: Edge, texture, shape features at conv-blocks 1-16\n\n"
        "[ENVIRONMENTAL INDICATORS]\n"
        "- ISC: Class-specific impervious surface cover estimate\n"
        "- NDVI/EVI range: Vegetation fraction and health indicator\n"
        "- LST anomaly: Thermal profile relative to rural reference surface\n\n"
        "[APPLICATIONS]\n"
        "LULC change detection | Urban planning support | "
        "Environmental impact assessment | Remote sensing classification benchmarking"
    )

all_classes = [
    'airplane','airport','baseball_diamond','basketball_court','beach','bridge','chaparral',
    'church','circular_farmland','cloud','commercial_area','dense_residential','desert',
    'forest','freeway','golf_course','ground_track_field','harbor','industrial_area',
    'intersection','island','lake','meadow','medium_residential','mobile_home_park',
    'mountain','overpass','palace','parking_lot','railway','railway_station',
    'rectangular_farmland','river','roundabout','runway','sea_ice','ship','snowberg',
    'sparse_residential','stadium','storage_tank','tennis_court','terrace',
    'thermal_power_station','wetland'
]

# Build new block using repr() so all special chars are safely escaped
lines = ['# Dictionary defining detailed technical analysis for land cover classes',
         'DETAILED_REPORTS = {']
for cls in all_classes:
    report = TECH.get(cls, general_tech(cls))
    lines.append('    ' + repr(cls) + ': ' + repr(report) + ',')
lines.append('}')
new_block = '\n'.join(lines)

# Find and replace block in app.py
pattern = r'# Dictionary defining detailed.*?DETAILED_REPORTS\s*=\s*\{.*?\n\}'
match = re.search(pattern, content, re.DOTALL)
if not match:
    pattern2 = r'DETAILED_REPORTS\s*=\s*\{.*?\n\}'
    match = re.search(pattern2, content, re.DOTALL)

if match:
    content = content[:match.start()] + new_block + content[match.end():]
    print("DETAILED_REPORTS: REPLACED")
else:
    print("ERROR: DETAILED_REPORTS block not found")

with open('app.py', 'w', encoding='utf-8') as f:
    f.write(content)
print("app.py saved OK")

# Verify syntax
import ast
try:
    ast.parse(content)
    print("Syntax check: PASSED")
except SyntaxError as e:
    print(f"Syntax error at line {e.lineno}: {e.msg}")
    lines = content.split('\n')
    s = max(0, e.lineno-3)
    for i, l in enumerate(lines[s:s+5], s+1):
        print(f"  {i}: {l[:100]}")
