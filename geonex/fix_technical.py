"""
Fix script:
1. Add CLAHE image preprocessing to app.py upload route
2. Replace DETAILED_REPORTS with structured technical reports
"""
import re

with open('app.py', 'r', encoding='utf-8') as f:
    content = f.read()

# -----------------------------------------------
# 1. Add cv2 import if not present
# -----------------------------------------------
if 'import cv2' not in content:
    content = content.replace('import numpy as np', 'import numpy as np\nimport cv2')
    print("cv2 import: ADDED")
else:
    print("cv2 import: already present")

# -----------------------------------------------
# 2. Add CLAHE preprocessing before model.predict
# -----------------------------------------------
old_predict = '''        img = image.load_img(local_path, target_size=IMG_SIZE)
        img_arr = image.img_to_array(img) / 255.0
        img_arr = np.expand_dims(img_arr, axis=0)

        preds = model.predict(img_arr)'''

new_predict = '''        img = image.load_img(local_path, target_size=IMG_SIZE)
        img_arr = image.img_to_array(img)

        # --- CLAHE Preprocessing: normalises contrast/brightness for real-world images ---
        img_bgr = cv2.cvtColor(img_arr.astype('uint8'), cv2.COLOR_RGB2BGR)
        img_lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(img_lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l_eq = clahe.apply(l)
        img_lab_eq = cv2.merge([l_eq, a, b])
        img_rgb_eq = cv2.cvtColor(cv2.cvtColor(img_lab_eq, cv2.COLOR_LAB2BGR), cv2.COLOR_BGR2RGB)
        img_arr = img_rgb_eq.astype('float32') / 255.0
        # ---------------------------------------------------------------------------------

        img_arr = np.expand_dims(img_arr, axis=0)
        preds = model.predict(img_arr)'''

if old_predict in content:
    content = content.replace(old_predict, new_predict)
    print("CLAHE preprocessing: ADDED")
else:
    print("WARNING: predict block not found - skipping CLAHE")

# -----------------------------------------------
# 3. Replace DETAILED_REPORTS with technical structured reports
# -----------------------------------------------
TECH = {
    "airplane": (
        "◆ SPECTRAL SIGNATURE\n"
        "• Dominant reflectance: High visible-band (VIS) albedo from metallic fuselage surfaces\n"
        "• NIR reflectance: High due to metal skin (aluminium/composite)\n"
        "• Thermal band: Low emissivity signature (metallic surface)\n"
        "• Shadow geometry: Elongated, asymmetric — diagnostic of fixed-wing aircraft\n\n"
        "◆ SPATIAL METRICS\n"
        "• Object dimensions (GSD-normalised): 20–80m length, 15–65m wingspan\n"
        "• Aspect ratio: ~1.3:1 (fuselage:wingspan)\n"
        "• Texture: Smooth, homogeneous surface with specular reflection\n"
        "• Shape descriptor: Bilateral symmetry — strong discriminator vs. ground vehicles\n\n"
        "◆ CLASSIFICATION FINDINGS\n"
        "• Class: Airplane (Object-level detection within airport/airfield context)\n"
        "• Confidence band: Derived from MobileNetV2 softmax probability output\n"
        "• Training set similarity: Spectral-spatial match to NWPU-RESISC45 'airplane' tiles (GSD: 0.2–30m)\n"
        "• Contextual cues matched: Tarmac surroundings, taxiway markings, apron proximity\n\n"
        "◆ ENVIRONMENTAL INDICATORS\n"
        "• Land Surface Temperature (LST): Elevated (reflective tarmac + engine heat)\n"
        "• Impervious Surface Cover (ISC): >95% in surrounding apron zone\n"
        "• NDVI: ~0.0 (no vegetation)\n"
        "• Noise pollution index: High (jet engine proximity zone)\n\n"
        "◆ GEOSPATIAL CONTEXT\n"
        "• Occurrence: Airport aprons, military airbases, cargo terminals, maintenance hangars\n"
        "• Associated infrastructure: Runways (GSD 30–60m width), taxiways, ATC towers\n"
        "• Risk indicators: High-value asset zone — security-sensitive\n\n"
        "◆ APPLICATIONS\n"
        "SAR/optical fusion for all-weather aircraft detection · Air traffic density monitoring · "
        "Airport capacity planning · Defense intelligence · Emergency airlift coordination"
    ),

    "airport": (
        "◆ SPECTRAL SIGNATURE\n"
        "• Runway surface: Grey asphalt — moderate reflectance across VIS bands (R≈0.15, G≈0.14, B≈0.13)\n"
        "• Terminal rooftops: High NIR reflectance (metal/concrete)\n"
        "• NDVI: ~0.0–0.05 (near-zero — dominantly impervious)\n"
        "• Thermal: Elevated LST across runway and apron surfaces\n\n"
        "◆ SPATIAL METRICS\n"
        "• Runway dimensions: 1,800–4,500m × 30–60m (type-dependent)\n"
        "• Terminal footprint: 10,000–200,000 m²\n"
        "• Runway orientation: Aligned with prevailing wind (ILS/VOR indicator)\n"
        "• Texture: Fine on runways (painted markings), coarse on surrounding zones\n\n"
        "◆ CLASSIFICATION FINDINGS\n"
        "• Class: Airport (facility-level land-use classification)\n"
        "• Key discriminators: Parallel runway geometry, apron-terminal adjacency, navigational aids\n"
        "• Training data match: High spatial frequency linear features + low NDVI background\n"
        "• False positive risk: Industrial zones (similar ISC) — differentiated by runway geometry\n\n"
        "◆ ENVIRONMENTAL INDICATORS\n"
        "• ISC: >80% — major impervious surface zone\n"
        "• NOx/VOC emissions: High (combustion point sources)\n"
        "• Stormwater runoff risk: High (glycol de-icing agents, jet fuel)\n"
        "• Urban Heat Island (UHI) contribution: Significant\n\n"
        "◆ GEOSPATIAL CONTEXT\n"
        "• Location typology: Peri-urban or exurban with highway access corridors\n"
        "• Buffer zone: Aviation safety height-restriction zone (6–13km radius)\n\n"
        "◆ APPLICATIONS\n"
        "Airport expansion monitoring · Runway capacity analysis · Noise contour mapping · "
        "Environmental compliance (ICAO Annex 16) · Emergency runway assessment"
    ),

    "beach": (
        "◆ SPECTRAL SIGNATURE\n"
        "• Sandy substrate: Very high VIS reflectance (R≈0.55, G≈0.50, B≈0.45)\n"
        "• Shoreline boundary: Sharp spectral contrast — sand (~0.5 reflectance) vs. water (~0.02–0.05)\n"
        "• NDWI (McFeeters): Positive in intertidal zone; negative on dry sand\n"
        "• NIR: Low (non-vegetated) — distinguishes from dune grasslands\n\n"
        "◆ SPATIAL METRICS\n"
        "• Shoreline geometry: Linear to curvilinear; sinuosity index 1.0–1.3\n"
        "• Beach width (GSD-relative): 5–300m\n"
        "• Texture: Low spatial frequency (smooth sand surface)\n"
        "• Spectral transition zone: Wet/dry sand boundary detectable at sub-5m GSD\n\n"
        "◆ CLASSIFICATION FINDINGS\n"
        "• Class: Beach (coastal sedimentary depositional environment)\n"
        "• Key discriminators: High-albedo narrow strip adjacent to dark water body\n"
        "• Confusion risk: Dry riverbeds (similar spectral) — differentiated by water adjacency\n"
        "• Training set similarity: Strong match to NWPU-RESISC45 beach class\n\n"
        "◆ ENVIRONMENTAL INDICATORS\n"
        "• Shoreline Change Rate: Detectable via multi-temporal NDWI analysis\n"
        "• Coastal Vulnerability Index (CVI): Moderate–High depending on exposure\n"
        "• Sea Level Rise sensitivity: High (low-elevation depositional zone)\n"
        "• Habitat Value: Critical nesting zone (Cheloniidae, Charadriiformes)\n\n"
        "◆ GEOSPATIAL CONTEXT\n"
        "• Coastal exposure: Wave energy determines sediment grain size and beach width\n"
        "• Risk: Storm surge inundation, coastal erosion, sand budget deficit\n\n"
        "◆ APPLICATIONS\n"
        "Shoreline change detection (DSAS) · Coastal erosion risk modelling · "
        "Marine protected area (MPA) delineation · Storm surge impact assessment · Tourism capacity analysis"
    ),

    "forest": (
        "◆ SPECTRAL SIGNATURE\n"
        "• NDVI: 0.65–0.90 (dense, healthy broadleaf or mixed canopy)\n"
        "• NIR reflectance: Very high (0.45–0.55) — cell structure scattering\n"
        "• Red band: Strong absorption (0.04–0.08) — chlorophyll a/b\n"
        "• EVI (Enhanced Vegetation Index): 0.4–0.7\n"
        "• NDWI (Gao): 0.1–0.4 (canopy water content)\n"
        "• Shortwave Infrared (SWIR): Moderate — canopy moisture indicator\n\n"
        "◆ SPATIAL METRICS\n"
        "• Canopy closure: >75%\n"
        "• Texture: High spatial variance — individual crown shadows create heterogeneity\n"
        "• Fractal dimension: 1.6–1.9 (complex edge geometry)\n"
        "• Crown size range (GSD-dependent): 3–25m diameter detectable\n\n"
        "◆ CLASSIFICATION FINDINGS\n"
        "• Class: Forest — Dense Canopy Land Cover (LCCS Class A1.1)\n"
        "• Confidence interpretation: High softmax probability = strong canopy signature\n"
        "• Training data match: Spectral-spatial alignment with NWPU-RESISC45 forest tiles\n"
        "• Confusion risk: Chaparral (lower NDVI), Meadow (no tree shadow texture)\n\n"
        "◆ ENVIRONMENTAL INDICATORS\n"
        "• Carbon Stock Estimate: 80–300 tC/ha (biome-dependent)\n"
        "• Above-Ground Biomass (AGB): High — requires SAR/LiDAR for precise estimation\n"
        "• LST: Suppressed (3–8°C cooler than surrounding non-vegetated land)\n"
        "• Evapotranspiration: High (latent heat flux dominant)\n"
        "• Fire Risk Index: Moderate (moisture-dependent)\n\n"
        "◆ GEOSPATIAL CONTEXT\n"
        "• Distribution: Tropical (0–23.5°N/S), Temperate (35–55°N/S), Boreal (50–70°N/S)\n"
        "• Deforestation indicators: Straight-edged clearings, road networks penetrating canopy\n\n"
        "◆ APPLICATIONS\n"
        "REDD+ carbon credit verification · Above-Ground Biomass mapping · "
        "Deforestation alert systems (GLAD, PRODES) · Wildfire fuel load assessment · "
        "Biodiversity hotspot identification · Watershed protection zoning"
    ),

    "river": (
        "◆ SPECTRAL SIGNATURE\n"
        "• Water reflectance: Very low VIS (R≈0.02–0.06) — strong absorption\n"
        "• NIR: Near-zero (strong water absorption) — primary water body discriminator\n"
        "• NDWI (McFeeters): Positive (>0.2) — reliable water detection index\n"
        "• Turbidity signature: Elevated red/green reflectance in sediment-laden water\n"
        "• Chlorophyll-a bloom: Green band anomaly in eutrophic conditions\n\n"
        "◆ SPATIAL METRICS\n"
        "• Channel width: Varies 5m–5km\n"
        "• Sinuosity index: 1.0 (straight) to >2.5 (meandering/anastomosing)\n"
        "• Braiding index: Multi-thread channels indicate high bedload transport\n"
        "• Floodplain extent: Detectable via NDWI time-series during flood events\n\n"
        "◆ CLASSIFICATION FINDINGS\n"
        "• Class: River (Linear inland water body — LCCS W1)\n"
        "• Key discriminators: Continuous dark linear feature with riparian vegetation fringe\n"
        "• Confusion risk: Road/shadow (similar geometry) — differentiated by NDWI positivity\n"
        "• Sinuosity pattern: Encodes geomorphic maturity and discharge regime\n\n"
        "◆ ENVIRONMENTAL INDICATORS\n"
        "• Sediment load (TSS): Quantifiable via red/NIR band ratio\n"
        "• Flood stage: NDWI expansion detectable vs. baseline water mask\n"
        "• Water quality anomalies: Algal bloom index, turbidity mapping\n"
        "• Riparian NDVI: Corridor health indicator (target: >0.4)\n\n"
        "◆ GEOSPATIAL CONTEXT\n"
        "• Drainage network: River order (Strahler) influences spectral width\n"
        "• Delta/estuary transition: Salinity gradient alters spectral properties\n\n"
        "◆ APPLICATIONS\n"
        "Flood plain inundation mapping · Water quality monitoring (TSS, Chl-a) · "
        "Hydropower siting · Drought stream-flow estimation · Aquatic habitat connectivity analysis"
    ),

    "desert": (
        "◆ SPECTRAL SIGNATURE\n"
        "• Bare soil reflectance: High VIS (R≈0.35–0.60, spectrally flat to red-shifted)\n"
        "• NDVI: <0.1 (near-zero or negative — bare mineral surface)\n"
        "• Albedo: 0.25–0.45 (sand) to 0.10–0.20 (dark reg/hamada)\n"
        "• Mineral composition: Iron oxide → red/orange hue; quartz → bright white\n"
        "• Thermal: Very high daytime LST (50–70°C surface temperature)\n\n"
        "◆ SPATIAL METRICS\n"
        "• Surface roughness: Low (smooth sheet sand/ergs) to high (rocky reg/yardangs)\n"
        "• Dune wavelength: 50m–2km (detectable at moderate GSD)\n"
        "• Texture: Directional (wind-formed dune patterns — NE orientation common)\n"
        "• Sand encroachment edge: Identifiable as advancing bright front\n\n"
        "◆ CLASSIFICATION FINDINGS\n"
        "• Class: Desert / Hyper-arid Zone (Köppen BWh/BWk)\n"
        "• Key discriminators: High albedo + near-zero NDVI + lack of structural features\n"
        "• Sub-types: Erg (dunes), Reg (gravel plain), Hammada (rocky plateau)\n"
        "• Training match: Spectrally bright, texturally smooth to moderate\n\n"
        "◆ ENVIRONMENTAL INDICATORS\n"
        "• Rainfall: <250mm/yr (hyper-arid: <25mm/yr)\n"
        "• Desertification Rate: Monitorable via NDVI trend analysis\n"
        "• Dust emission potential: High (Aeolian sediment source)\n"
        "• LST diurnal range: 30–50°C (large thermal inertia contrast)\n\n"
        "◆ GEOSPATIAL CONTEXT\n"
        "• Geographic distribution: Sahara, Arabian Peninsula, Atacama, Gobi, Thar\n"
        "• Encroachment threat: Semi-arid Sahel transition zone most at risk\n\n"
        "◆ APPLICATIONS\n"
        "Desertification monitoring (UNCCD) · Renewable energy (CSP/PV) site selection · "
        "Mineral exploration · Sand dune migration modelling · Dust storm source identification"
    ),

    "mountain": (
        "◆ SPECTRAL SIGNATURE\n"
        "• Exposed rock: Variable reflectance (0.10–0.45) — lithology dependent\n"
        "• Snow/ice cap: Very high albedo (0.60–0.90)\n"
        "• Vegetation belts: NDVI gradient (0.6–0.8 montane forest → 0.2–0.4 alpine meadow → ~0 barren)\n"
        "• Shadow fraction: High (illumination geometry at high relief)\n"
        "• Thermal: Low LST at elevation (lapse rate: ~6.5°C/1000m)\n\n"
        "◆ SPATIAL METRICS\n"
        "• Relief amplitude: Hundreds to thousands of metres\n"
        "• Slope gradient: Detectable via shadow azimuth and DEM derivatives\n"
        "• Texture: High spatial variance (heterogeneous surface)\n"
        "• Ridge/valley pattern: Crenulated boundary geometry — high fractal dimension\n\n"
        "◆ CLASSIFICATION FINDINGS\n"
        "• Class: Mountain / Highland Terrain (Complex terrain LCCS)\n"
        "• Key discriminators: Shadow gradient, snow cap, vegetation zonation pattern\n"
        "• Altitudinal zones identifiable: Montane / Subalpine / Alpine / Nival\n"
        "• Confusion risk: Chaparral at foothills — differentiated by shadow intensity\n\n"
        "◆ ENVIRONMENTAL INDICATORS\n"
        "• Snow Cover Area (SCA): Seasonal change detectable via NDSI index\n"
        "• Glacier mass balance: Terminus retreat measurable via multi-temporal imagery\n"
        "• Landslide susceptibility: High on steep scarps (slope >30°)\n"
        "• Permafrost presence: Detectable via thermal anomaly mapping\n\n"
        "◆ GEOSPATIAL CONTEXT\n"
        "• Water tower function: Snow/glacier melt supplies 1.9B people\n"
        "• Tectonic setting: Active vs. passive margin determines geomorphic stage\n\n"
        "◆ APPLICATIONS\n"
        "Glacier mass balance (GRACE/ICESat) · Snowmelt runoff modelling · "
        "Landslide hazard mapping · Alpine biodiversity surveys · Avalanche risk zoning"
    ),

    "lake": (
        "◆ SPECTRAL SIGNATURE\n"
        "• Open water: Very low NIR reflectance (<0.05) — strong absorption\n"
        "• NDWI: >0.3 (clean freshwater); reduced in turbid/eutrophic conditions\n"
        "• Turbidity: Elevated red/green ratio — sediment load indicator\n"
        "• Cyanobacterial bloom: NIR shoulder at 700nm — detectable via MCI index\n"
        "• Depth indicator: Blue/green ratio (clearer = higher blue ratio)\n\n"
        "◆ SPATIAL METRICS\n"
        "• Shape index: Varies from circular (volcanic crater) to elongate (glacial/tectonic)\n"
        "• Shoreline sinuosity: 1.0–2.5\n"
        "• Surface area: Directly measurable via NDWI thresholding\n"
        "• Fetch length: Key for wind-wave energy estimation\n\n"
        "◆ CLASSIFICATION FINDINGS\n"
        "• Class: Inland Water Body — Lake (LCCS W1.1)\n"
        "• Key discriminators: Enclosed dark water mass with defined land boundary\n"
        "• Sub-types: Freshwater, Saline (higher blue/green), Seasonal (NDWI time-series)\n"
        "• Training match: Dark enclosed polygon distinct from river linear features\n\n"
        "◆ ENVIRONMENTAL INDICATORS\n"
        "• Trophic state: Oligotrophic → Eutrophic (NDVI/Chl-a proxies)\n"
        "• Water volume change: Surface area × bathymetry (altimetry-derived)\n"
        "• Thermal stratification: LST gradient across lake surface\n"
        "• Drought sensitivity: High — surface area reduction detectable in multi-temporal analysis\n\n"
        "◆ APPLICATIONS\n"
        "Water quality monitoring (WHO compliance) · Reservoir storage estimation · "
        "Algal bloom early warning · Drought/flood impact assessment · Ramsar wetland status"
    ),

    "dense_residential": (
        "◆ SPECTRAL SIGNATURE\n"
        "• Rooftop mix: Red tile (high R, low B), asphalt (low VIS), concrete (flat moderate VIS)\n"
        "• NDVI: Very low (0.0–0.15) — minimal vegetation between structures\n"
        "• ISC (Impervious Surface Cover): >75%\n"
        "• Thermal: High LST (UHI hotspot) — 4–8°C above suburban fringe\n"
        "• Shadow fraction: High (multi-storey structures, narrow lanes)\n\n"
        "◆ SPATIAL METRICS\n"
        "• Building density: >60 units/ha\n"
        "• Plot coverage ratio: 0.7–0.9\n"
        "• Road network width: 3–8m (narrow lanes)\n"
        "• Texture: Fine-grain, high spatial frequency\n"
        "• Building footprint homogeneity: High (similar size within neighbourhood)\n\n"
        "◆ CLASSIFICATION FINDINGS\n"
        "• Class: Dense Residential (Urban LCCS — Built-up B1.1)\n"
        "• Key discriminators: High rooftop density + fine texture + low NDVI\n"
        "• Differentiated from commercial: Smaller footprint size, denser packing\n"
        "• Morphological type: Organic grid (informal) vs. regular grid (planned)\n\n"
        "◆ ENVIRONMENTAL INDICATORS\n"
        "• UHI Intensity: High (>4°C above rural reference)\n"
        "• Green Space Ratio: <10% (below WHO 9m²/person recommendation)\n"
        "• Population density proxy: High (>10,000 persons/km²)\n"
        "• Flood runoff coefficient: 0.7–0.9 (high imperviousness)\n\n"
        "◆ APPLICATIONS\n"
        "Population density modelling · Slum/informal settlement mapping · "
        "UHI mitigation planning · Infrastructure demand analysis · Census augmentation"
    ),

    "commercial_area": (
        "◆ SPECTRAL SIGNATURE\n"
        "• Flat rooftops: Bright NIR (metal/EPDM) or moderate VIS (concrete/gravel ballast)\n"
        "• NDVI: 0.0–0.1 (near-zero vegetation)\n"
        "• ISC: >85% (near-complete imperviousness)\n"
        "• Thermal: Very high daytime LST (reflective/absorptive flat roofs + HVAC waste heat)\n"
        "• Parking surfaces: Dark asphalt — low VIS reflectance (0.05–0.10)\n\n"
        "◆ SPATIAL METRICS\n"
        "• Building footprint: Large (1,000–50,000 m²)\n"
        "• Roof-to-ground ratio: 0.5–0.8\n"
        "• Texture: Coarse within buildings, fine in parking lots\n"
        "• Road network: Wide arterials (8–30m) with loading bays\n\n"
        "◆ CLASSIFICATION FINDINGS\n"
        "• Class: Commercial / Retail Zone (Urban Built-up B2)\n"
        "• Key discriminators: Large homogeneous rooftops + expansive parking surfaces + road access\n"
        "• False positive: Industrial (similar ISC) — differentiated by loading dock geometry\n\n"
        "◆ ENVIRONMENTAL INDICATORS\n"
        "• LST: 5–10°C above vegetated land (peak afternoon)\n"
        "• CO₂ equivalent: High energy consumption per m² (HVAC, lighting)\n"
        "• Stormwater: Near-total impervious runoff (>0.85 runoff coefficient)\n\n"
        "◆ APPLICATIONS\n"
        "Retail trade area mapping · UHI hotspot analysis · "
        "EV charging infrastructure siting · Commercial property assessment"
    ),

    "industrial_area": (
        "◆ SPECTRAL SIGNATURE\n"
        "• Metal roofs: Very high NIR reflectance (specular) + moderate VIS\n"
        "• Stockpile materials: Spectral mix of coal (black), limestone (white), ore (brown)\n"
        "• NDVI: ~0.0 (no vegetation in active industrial zone)\n"
        "• Thermal: Elevated point sources (kiln, furnace, cooling towers)\n"
        "• Smoke plumes: Detectable in TIR/SWIR bands\n\n"
        "◆ SPATIAL METRICS\n"
        "• Building footprint: Very large (5,000–200,000 m²), low pitch\n"
        "• ISC: >80%\n"
        "• Road access: Wide (HGV routing), rail siding present in heavy industry\n"
        "• Chimney shadows: Point source indicators — height estimable from shadow length\n\n"
        "◆ CLASSIFICATION FINDINGS\n"
        "• Class: Industrial / Manufacturing Zone (B3 — LCCS)\n"
        "• Key discriminators: Heterogeneous large-footprint buildings + material stockpiles + rail/road access\n"
        "• Pollution pressure: Chimney plume direction indicates prevailing wind\n\n"
        "◆ ENVIRONMENTAL INDICATORS\n"
        "• Air pollution (PM2.5, SO₂, NOx): detectable via Sentinel-5P TROPOMI\n"
        "• Groundwater contamination risk: High (heavy metals, solvents)\n"
        "• LST anomaly: Point source thermal signature detectable in TIRS\n\n"
        "◆ APPLICATIONS\n"
        "Industrial emission source attribution · Environmental compliance monitoring · "
        "Supply chain infrastructure mapping · Land contamination risk screening"
    ),

    "wetland": (
        "◆ SPECTRAL SIGNATURE\n"
        "• Saturated soil: Low NIR, moderate VIS — darker than dry soil\n"
        "• Emergent vegetation: Moderate NDVI (0.3–0.6) — lower than upland forest\n"
        "• Open water patches: NDWI positive (>0.2)\n"
        "• NDWI/NDVI hybrid: Heterogeneous spatial pattern — diagnostic of wetland mosaic\n"
        "• Peat/organic soil: Very low reflectance across all bands (darkening effect)\n\n"
        "◆ SPATIAL METRICS\n"
        "• Surface heterogeneity: High — interspersed open water, emergent veg, mud\n"
        "• Texture: Mixed fine–coarse depending on vegetation density\n"
        "• Boundary geometry: Diffuse (gradational) rather than sharp\n"
        "• Seasonal extent change: NDWI/NDWI time-series essential for full mapping\n\n"
        "◆ CLASSIFICATION FINDINGS\n"
        "• Class: Wetland (LCCS W2 — Inland non-permanent/permanent)\n"
        "• Sub-types: Marsh, Swamp, Mangrove (coastal), Peatland, Floodplain\n"
        "• Key discriminators: Hybrid NDVI/NDWI signature + seasonal inundation pattern\n"
        "• Confusion risk: Rice paddy (similar signal) — SAR-C backscatter can differentiate\n\n"
        "◆ ENVIRONMENTAL INDICATORS\n"
        "• Blue Carbon: Mangroves/peatlands store 3–5× forest carbon per unit area\n"
        "• Biodiversity Index: Very High — >35% of all species depend on wetlands\n"
        "• Flood regulation service: Peak flow attenuation capacity\n"
        "• Methane (CH₄) emissions: Significant (anaerobic decomposition)\n\n"
        "◆ APPLICATIONS\n"
        "Ramsar Convention site monitoring · Blue carbon accounting · "
        "Flood storage capacity mapping · Biodiversity conservation prioritisation · "
        "Water purification service valuation"
    ),
}

# General template for classes not in TECH dict
def general_tech(cls):
    name = cls.replace('_', ' ').title()
    return (
        f"◆ SPECTRAL SIGNATURE\n"
        f"• VIS/NIR reflectance: Class-specific spectral profile matched to MobileNetV2 training features\n"
        f"• NDVI: Dependent on vegetation fraction within class extent\n"
        f"• Thermal (LST): Varies by surface material and land-use activity level\n\n"
        f"◆ SPATIAL METRICS\n"
        f"• Texture: Characteristic spatial frequency and pattern for {name}\n"
        f"• Geometry: Distinctive shape descriptors used in CNN feature extraction\n"
        f"• Object scale: Consistent with training patch GSD (0.2–30m ground resolution)\n\n"
        f"◆ CLASSIFICATION FINDINGS\n"
        f"• Class: {name} (LCCS-compliant land cover category)\n"
        f"• Confidence: Derived from MobileNetV2 final softmax layer probability output\n"
        f"• Training similarity: Spectral-spatial feature match to NWPU-RESISC45 / AID dataset {name} tiles\n"
        f"• CNN feature layers: Edge, texture, shape features extracted at conv-block 1–16\n\n"
        f"◆ ENVIRONMENTAL INDICATORS\n"
        f"• ISC (Impervious Surface Cover): Class-specific estimate\n"
        f"• NDVI/EVI range: Vegetation fraction indicator\n"
        f"• LST anomaly: Thermal profile relative to rural reference surface\n\n"
        f"◆ GEOSPATIAL CONTEXT\n"
        f"• Spatial distribution: Characteristic of specific climate, topographic, and socioeconomic contexts\n"
        f"• Adjacent land uses: Typical neighbourhood land-use relationships\n\n"
        f"◆ APPLICATIONS\n"
        f"Land-use/land-cover (LULC) change detection · Urban planning support · "
        f"Environmental impact assessment · Remote sensing classification benchmarking"
    )

# Build full new DETAILED_REPORTS string
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

lines = ['# Dictionary defining detailed technical analysis for land cover classes\nDETAILED_REPORTS = {']
for cls in all_classes:
    report = TECH.get(cls, general_tech(cls))
    # Escape backslashes and quotes
    report_escaped = report.replace('\\', '\\\\').replace('"', '\\"')
    lines.append(f'    "{cls}": "{report_escaped}",')
lines.append('}')
new_detailed_reports = '\n'.join(lines)

# Find and replace the DETAILED_REPORTS section in app.py
pattern = r'# Dictionary defining detailed(?:.*?)technical.*?analysis.*?DETAILED_REPORTS\s*=\s*\{.*?\n\}'
match = re.search(pattern, content, re.DOTALL)
if not match:
    # Try alternate patterns
    pattern2 = r'DETAILED_REPORTS\s*=\s*\{.*?\n\}'
    match = re.search(pattern2, content, re.DOTALL)

if match:
    content = content[:match.start()] + new_detailed_reports + content[match.end():]
    print("DETAILED_REPORTS: REPLACED with technical structured reports")
else:
    print("ERROR: Could not find DETAILED_REPORTS in app.py")

with open('app.py', 'w', encoding='utf-8') as f:
    f.write(content)
print("app.py saved OK")
