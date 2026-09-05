"""
GeoTips-derived CLIP prompts, one short caption per visual cue.

WHY IT IS STRUCTURED THIS WAY
CLIP's text encoder truncates at 77 tokens. A single paragraph stuffing
every clue about a country would be silently cut off mid-sentence, so
each country instead gets several SHORT captions - one per cue family -
and their embeddings are averaged. That is standard prompt ensembling,
and it keeps every clue inside the token budget.

WHERE THIS IS USED
v7 classifies with a trained head on CLIP *image* features, so these
prompts do not affect its accuracy on their own. They are used for:
  1. the zero-shot baseline (what CLIP scores with no training)
  2. an optional text-anchor ensemble fused with the trained head,
     which the notebook measures rather than assumes

SOURCING
France and Ireland entries were read from geotips.net directly. The rest
come from general GeoGuessr meta knowledge in the same style; the site
keeps all countries on one large page that could not be paged through
in full. Treat individual details as good-but-unverified, and prefer
whatever the measured numbers say.
"""

# Cue families: bollard, lines, plates, poles, signs, arch, flora, misc
GEOTIPS = {
'Albania': {
  'arch': 'unfinished red brick and concrete houses with exposed rebar',
  'flora': 'dry mediterranean scrub on steep limestone hills',
  'signs': 'Albanian language signs, SH road numbers',
  'misc': 'many old Mercedes cars, older low-quality camera coverage',
  'lines': 'worn faded road markings on narrow patched tarmac'},
'Andorra': {
  'arch': 'stone and slate Pyrenean buildings with steep roofs',
  'flora': 'high alpine pine slopes and bare rock above the treeline',
  'signs': 'Catalan language signs, CG road numbers',
  'lines': 'white lines on winding mountain switchback roads',
  'misc': 'dense hillside apartment blocks in narrow valleys'},
'Austria': {
  'arch': 'tidy alpine houses with wide eaves and painted facades',
  'flora': 'steep spruce-covered alpine slopes and green meadows',
  'signs': 'German language signs, red and white striped markers',
  'bollard': 'grey and white reflector posts along mountain roads',
  'lines': 'crisp white centre lines on well maintained tarmac',
  'plates': 'white plates with a red and white stripe'},
'Belgium': {
  'arch': 'narrow brick row houses with varied rooflines',
  'flora': 'flat farmland with poplar rows and small woods',
  'signs': 'Dutch and French bilingual signs',
  'lines': 'concrete slab roads with visible joins and worn markings',
  'misc': 'motorways lit by long rows of orange street lights',
  'bollard': 'square white concrete bollards'},
'Bosnia and Herzegovina': {
  'arch': 'red tiled roofs, unfinished concrete, minarets and churches',
  'flora': 'steep green wooded hills and karst river valleys',
  'signs': 'Bosnian signs mixing Latin and Cyrillic script',
  'misc': 'older lower quality camera coverage, narrow patched roads',
  'lines': 'faded white lines, frequent guardrails on gorge roads'},
'Bulgaria': {
  'arch': 'communist era concrete blocks and red roofed village houses',
  'flora': 'rolling farmland, oak woods and bare limestone ridges',
  'signs': 'Cyrillic script road signs in Bulgarian',
  'bollard': 'black and white striped kerbstones and posts',
  'misc': 'older camera coverage, worn rural tarmac'},
'Croatia': {
  'arch': 'white limestone houses with orange terracotta roofs',
  'flora': 'karst scrub, cypress and pine near the Adriatic coast',
  'signs': 'Croatian language signs, D road numbers',
  'lines': 'white lines with stone walls lining narrow coastal roads',
  'plates': 'plates with a red and white chequered shield'},
'Cyprus': {
  'arch': 'flat roofed concrete houses with solar water tanks',
  'flora': 'dry yellow hills with olive and carob trees',
  'signs': 'Greek and English bilingual signs',
  'misc': 'traffic drives on the left, British style road furniture',
  'lines': 'double yellow lines at the kerb, wide chip seal roads'},
'Czech Republic': {
  'arch': 'pastel village houses with steep orange tiled roofs',
  'flora': 'rolling farmland with spruce plantations and hop fields',
  'signs': 'Czech language signs with hacek and acute accents',
  'bollard': 'white bollards with an orange reflector strip',
  'lines': 'white lines on grey tarmac, avenues of roadside trees',
  'misc': 'red and white striped guardrail ends'},
'Denmark': {
  'arch': 'red brick houses with white window frames, low and wide',
  'flora': 'very flat farmland, wind turbines and trimmed hedgerows',
  'signs': 'Danish language signs with the letters ae, oe and aa',
  'lines': 'white lines with wide separate cycle paths beside roads',
  'bollard': 'slim white posts with red reflectors'},
'Estonia': {
  'arch': 'wooden houses with steep roofs and Soviet era blocks',
  'flora': 'flat boreal pine and birch forest, bogs and reeds',
  'signs': 'Estonian language signs with umlauts and double vowels',
  'lines': 'white lines on narrow roads, long straight forest corridors',
  'misc': 'tall snow poles and reflector posts at the verge'},
'Finland': {
  'arch': 'wooden houses painted red or yellow among trees',
  'flora': 'dense boreal spruce and pine forest with lakes',
  'signs': 'Finnish and Swedish bilingual signs with double letters',
  'lines': 'white lines, wide gravel shoulders and reflector posts',
  'misc': 'frequent gravel roads through unbroken conifer forest'},
# --- read directly from geotips.net ---
'France': {
  'bollard': 'large white bollards, some with a red reflector band',
  'lines': 'dashed white shoulder lines and diamond shaped dashes',
  'poles': 'wooden utility poles marked with a small blue square sticker',
  'signs': 'blue and green arrow direction signs, street names using Rue',
  'arch': 'pale render houses with shutters and slate or tile roofs',
  'flora': 'plane trees lining roads, farmland and mixed woodland',
  'plates': 'white EU plates, older coverage with yellow rear plates'},
'Germany': {
  'arch': 'half timbered houses and tidy render homes with grey roofs',
  'flora': 'dense mixed farmland, beech woods and spruce plantations',
  'signs': 'German language signs, dense signage and clear white lettering',
  'bollard': 'white posts with black reflector tops',
  'lines': 'crisp white lines on smooth well maintained tarmac',
  'misc': 'red and white striped kerb posts, immaculate road surfaces'},
'Greece': {
  'arch': 'whitewashed flat roofed houses with concrete pillars',
  'flora': 'olive groves, dry scrub hills and cypress trees',
  'signs': 'Greek alphabet road signs, often with Latin transliteration',
  'lines': 'faded white lines on narrow winding roads with low walls',
  'misc': 'roadside shrines, unfinished concrete frames, dry stone walls'},
'Hungary': {
  'arch': 'single storey farmhouses gable end to the road',
  'flora': 'very flat plains, maize and sunflower fields, poplar rows',
  'signs': 'Hungarian language signs with double acute accents',
  'bollard': 'black and white striped kerbstones',
  'lines': 'white lines on flat straight roads with tree avenues'},
'Iceland': {
  'arch': 'corrugated metal clad houses, no trees around them',
  'flora': 'treeless volcanic terrain, moss, black lava and grass',
  'signs': 'Icelandic language signs with thorn and eth letters',
  'lines': 'yellow centre lines and yellow topped reflector posts',
  'misc': 'gravel roads, distant glaciers, wide empty basalt landscapes'},
# --- read directly from geotips.net ---
'Ireland': {
  'bollard': 'chunky green bollards, relatively rare',
  'lines': 'dashed yellow side lines and white centre markings',
  'plates': 'standard EU style plates',
  'poles': 'wooden utility poles throughout',
  'signs': 'English with Irish translations, green white and brown signs',
  'arch': 'boxy houses with the front door in the centre',
  'misc': 'traffic drives on the left, distances in kilometres'},
'Italy': {
  'arch': 'terracotta roofed houses with ochre and cream render',
  'flora': 'cypress trees, olive groves, vineyards and dry hills',
  'signs': 'Italian language signs, SP and SS road numbers',
  'bollard': 'white bollards with a black band near the top',
  'lines': 'white lines on narrow roads walled by stone or hedge',
  'misc': 'roadside shrines, tightly packed hilltop villages'},
'Latvia': {
  'arch': 'wooden farmhouses and Soviet era apartment blocks',
  'flora': 'flat pine and birch forest with wet meadows',
  'signs': 'Latvian language signs with macrons and cedillas',
  'lines': 'white lines on narrow roads, many unpaved gravel stretches',
  'bollard': 'black and white striped posts at the verge'},
'Liechtenstein': {
  'arch': 'immaculate alpine chalets with wide balconies',
  'flora': 'steep wooded alpine valley slopes above the Rhine',
  'signs': 'German language signs in Swiss style',
  'lines': 'flawless white lines on very clean narrow roads',
  'misc': 'dense tidy villages against sudden steep mountain walls'},
'Lithuania': {
  'arch': 'wooden village houses and low brick farm buildings',
  'flora': 'flat farmland with birch stands and pine forest',
  'signs': 'Lithuanian language signs with ogoneks and caron letters',
  'lines': 'white lines on flat straight roads with wide verges',
  'bollard': 'white posts with red reflectors along the shoulder'},
'Luxembourg': {
  'arch': 'prosperous stone and render houses with slate roofs',
  'flora': 'wooded rolling hills and small tidy pastures',
  'signs': 'French and German signs, CR and N road numbers',
  'lines': 'white lines on smooth well kept narrow roads',
  'misc': 'dense forest valleys with very tidy small villages'},
'Malta': {
  'arch': 'honey coloured limestone buildings with flat roofs',
  'flora': 'dry rocky terrain, prickly pear and low stone walls',
  'signs': 'Maltese and English bilingual signs',
  'misc': 'traffic drives on the left, very narrow walled lanes',
  'lines': 'worn markings, dry stone walls right at the road edge'},
'Monaco': {
  'arch': 'dense modern high rise apartments on a steep coast',
  'flora': 'palms and manicured mediterranean planting',
  'signs': 'French language signs in a dense urban setting',
  'misc': 'luxury cars, tunnels, multi level roads above the sea',
  'lines': 'immaculate white lines on steep urban switchbacks'},
'Montenegro': {
  'arch': 'stone houses with orange roofs, unfinished concrete',
  'flora': 'dramatic bare karst mountains meeting the Adriatic',
  'signs': 'Montenegrin signs in Latin and Cyrillic script',
  'lines': 'narrow roads with frequent guardrails above steep drops',
  'misc': 'older camera coverage, tunnels cut through bare rock'},
'Netherlands': {
  'arch': 'brick houses with large windows in neat terraces',
  'flora': 'very flat polder farmland, drainage ditches, few hills',
  'signs': 'Dutch language signs, N road numbers',
  'lines': 'red asphalt cycle paths beside white lined roads',
  'misc': 'canals, dykes, huge numbers of bicycles, wind turbines'},
'North Macedonia': {
  'arch': 'red roofed houses and unfinished concrete on hillsides',
  'flora': 'dry scrub hills, tobacco and vine fields in valleys',
  'signs': 'Macedonian Cyrillic road signs',
  'misc': 'older lower quality camera coverage, patched narrow roads',
  'lines': 'faded white lines, black and white striped kerbstones'},
'Norway': {
  'arch': 'red and white painted wooden houses with grass roofs',
  'flora': 'steep spruce forest, bare fells, fjords and waterfalls',
  'signs': 'Norwegian language signs with the letters ae, oe and aa',
  'lines': 'yellow centre lines and white edge lines',
  'misc': 'frequent long tunnels, rock cuttings, guardrails on fjord roads'},
'Poland': {
  'arch': 'square render houses with steep roofs, brick farm sheds',
  'flora': 'flat farmland in long narrow strips, birch and pine',
  'signs': 'Polish language signs with slashed l and tailed letters',
  'poles': 'concrete utility poles with distinctive cross arms',
  'lines': 'white lines on narrow roads with tree avenues',
  'bollard': 'white posts with red reflector bands'},
'Portugal': {
  'arch': 'white houses with orange roofs and azulejo tiled walls',
  'flora': 'eucalyptus and cork oak plantations, pine near the coast',
  'signs': 'Portuguese language signs, N and EN road numbers',
  'lines': 'white lines on dark tarmac, cobbled village streets',
  'misc': 'black and white cobbled pavements, many eucalyptus stands'},
'Romania': {
  'arch': 'colourful village houses with ornate metal gates',
  'flora': 'Carpathian beech and spruce forest, small hay meadows',
  'signs': 'Romanian language signs with comma below letters',
  'misc': 'orthodox churches, horse carts, older camera coverage',
  'bollard': 'black and white striped kerbstones and posts'},
'Serbia': {
  'arch': 'red roofed brick houses, unfinished upper storeys',
  'flora': 'flat Pannonian farmland in the north, wooded hills south',
  'signs': 'Serbian signs in both Cyrillic and Latin script',
  'bollard': 'black and white striped kerbstones',
  'misc': 'older camera coverage, wide flat maize fields'},
'Slovakia': {
  'arch': 'painted village houses with steep tiled roofs',
  'flora': 'Carpathian spruce forest and narrow farmed valleys',
  'signs': 'Slovak language signs with caron and acute accents',
  'bollard': 'white bollards with an orange reflector strip',
  'lines': 'white lines on narrow valley roads with tree avenues',
  'misc': 'wooded ridges rising directly behind small villages'},
'Slovenia': {
  'arch': 'alpine houses with wide eaves and white church spires',
  'flora': 'steep green forested hills, vineyards and karst meadows',
  'signs': 'Slovenian language signs with caron letters',
  'lines': 'crisp white lines on very well maintained narrow roads',
  'misc': 'hayracks in fields, onion domed church towers on hilltops'},
'Spain': {
  'arch': 'white or ochre houses with flat terracotta roofs',
  'flora': 'olive groves, dry meseta scrub and irrigated orchards',
  'signs': 'Spanish language signs, N and A road numbers',
  'lines': 'white lines on wide smooth roads through dry country',
  'misc': 'dry stone terraces, dusty verges, sparse holm oak dehesa'},
'Sweden': {
  'arch': 'falu red wooden houses with white trim',
  'flora': 'pine and spruce forest, birch stands, lakes and bogs',
  'signs': 'Swedish language signs with the letters aa, ae and oe',
  'lines': 'white lines on wide roads with tall reflector posts',
  'misc': 'long straight forest roads, rock cuttings, red barns'},
'Switzerland': {
  'arch': 'wooden chalets with wide balconies and stone farmhouses',
  'flora': 'steep alpine meadows, spruce forest and bare high peaks',
  'signs': 'German French or Italian signs, immaculate signage',
  'lines': 'flawless white lines on pristine narrow mountain roads',
  'misc': 'very clean road furniture, cable cars, hillside cow pasture'},
'Turkey': {
  'arch': 'concrete apartment blocks with solar water heaters',
  'flora': 'dry Anatolian steppe, poplar rows and irrigated fields',
  'signs': 'Turkish language signs with dotless i and cedilla letters',
  'misc': 'mosques with slim minarets, frequent red Turkish flags',
  'lines': 'white lines on wide roads, dusty gravel shoulders'},
'Ukraine': {
  'arch': 'Soviet era blocks and small render cottages',
  'flora': 'vast flat black earth farmland with poplar windbreaks',
  'signs': 'Ukrainian Cyrillic road signs',
  'poles': 'concrete utility poles along wide open plains',
  'misc': 'long straight roads, sunflower fields, worn tarmac'},
'United Kingdom': {
  'arch': 'brick terraced houses with slate roofs and chimney stacks',
  'flora': 'trimmed hedgerows, small green fields and oak trees',
  'signs': 'English language signs, distances in miles',
  'misc': 'traffic drives on the left, roundabouts and yellow kerb lines',
  'lines': 'white lines on narrow lanes enclosed by high hedges'},
}

TEMPLATE = 'a Google Street View photo taken in {c}, {cue}'
PLAIN    = 'a Google Street View photo taken in {c}'


def country_captions(country):
    """Several short captions per country, each within CLIP's 77 tokens."""
    caps = [PLAIN.format(c=country)]
    for cue in GEOTIPS.get(country, {}).values():
        caps.append(TEMPLATE.format(c=country, cue=cue))
    return caps


def plain_captions(country):
    """The simple baseline, for measuring what the rich prompts add."""
    return [PLAIN.format(c=country)]
