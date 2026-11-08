class_names_isaid = ['ship','storage tank', 'baseball diamond', 'tennis court', 'basketball court', 
                     'ground track field', 'bridge', 'large vehicle', 'small vehicle', 'helicopter',
                     'swimming pool', 'roundabout', 'soccer ball field', 'plane', 'harbor'
]

BACKGROUND_CATEGORY=["background", "other region","without any objects"]


class_names_potsdam = ['impervious surfaces', 'building', 'low vegetation', 'tree','car']

BACKGROUND_CATEGORY_potsdam =['background', 'clutter']

class_descriptions_isaid = {
    "ship": [
        "a high-resolution overhead view of a large ship on open water in a satellite image",
        "aerial view of a single vessel far from the shore in remote sensing imagery",
        "satellite image showing a cargo ship at sea with no nearby harbor",
        "overhead view of a moving ship creating a wake in the water",
        "remote sensing view of a large ship isolated from docks or piers",
        "aerial photo of a single vessel in deep ocean waters"
    ],
    "storage tank": [
        "aerial view of a large circular storage tank in an industrial area",
        "an overhead view of a fuel or oil tank in satellite imagery",
        "remote sensing image of a round above-ground storage tank",
        "satellite image showing a group of petroleum storage tanks",
        "overhead view of a cylindrical tank structure in a tank farm",
        "aerial photo of an oil or chemical storage facility from above"
    ],
    "baseball diamond": [
        "overhead view of a baseball diamond sports field with base lines",
        "satellite image showing a diamond-shaped baseball field",
        "aerial view of a baseball field with grass and dirt areas",
        "remote sensing view of a baseball stadium from above",
        "aerial image showing a baseball diamond in a sports complex",
        "satellite view of a community baseball field in an open area"
    ],
    "tennis court": [
        "an overhead view of a rectangular tennis court with clear markings",
        "aerial image of a tennis court in remote sensing imagery",
        "satellite view of a tennis playing field with white boundary lines",
        "remote sensing view of multiple tennis courts in an urban area",
        "aerial photo of an outdoor tennis court with green surface",
        "satellite image showing a tennis court in a sports facility"
    ],
    "basketball court": [
        "overhead view of a basketball court with visible hoops and markings",
        "aerial image of a painted basketball court in urban surroundings",
        "satellite image showing a small rectangular basketball court",
        "remote sensing image of a basketball court in a park area",
        "aerial photo of multiple basketball courts side by side",
        "satellite view of an outdoor basketball court with clear lines"
    ],
    "ground track field": [
        "overhead view of an athletics running track surrounding a grass field",
        "aerial image of an oval track field in satellite imagery",
        "remote sensing view of a stadium track and field facility",
        "satellite photo of a sports track with lanes and a central field",
        "aerial view of a running track in a school sports complex",
        "geospatial image showing a red oval track with green infield"
    ],
    "bridge": [
        "overhead view of a bridge crossing over water or roads in satellite imagery",
        "aerial image of a transportation bridge structure",
        "satellite view of a highway or railway bridge spanning a river",
        "remote sensing image of a long bridge connecting two land masses",
        "aerial photo of a bridge over a large body of water",
        "satellite image of a multi-lane road bridge in an urban area"
    ],
    "large vehicle": [
        "overhead view of a large vehicle such as a truck or bus in satellite imagery",
        "aerial image of a heavy-duty vehicle on a road or parking lot",
        "satellite image showing a commercial freight vehicle from above",
        "remote sensing image of a large truck on a highway",
        "aerial photo of a bus in a transit terminal from above",
        "satellite view of multiple large vehicles parked in an open lot"
    ],
    "small vehicle": [
        "a high-resolution overhead view of a small car or van on a road",
        "satellite image showing a compact vehicle in an open parking area",
        "aerial photo of a single small vehicle on an empty street",
        "remote sensing image of a tiny vehicle in a large urban scene",
        "overhead view of one passenger car in a parking lot",
        "satellite image of a small vehicle isolated from other cars"
    ],
    "helicopter": [
        "overhead view of a helicopter or rotorcraft in remote sensing imagery",
        "aerial image showing a helicopter with visible rotor blades",
        "satellite view of a helicopter on a helipad or runway",
        "remote sensing photo of a helicopter taking off from an airfield",
        "aerial view of a parked helicopter in an open area",
        "satellite image showing multiple helicopters on a tarmac"
    ],
    "swimming pool": [
         "a high-resolution overhead view of a helicopter in an open airfield",
        "satellite image showing a single helicopter with rotor blades visible",
        "aerial view of a helicopter parked on a helipad",
        "remote sensing image of one rotorcraft in a clear open area",
        "overhead photo of a helicopter isolated from buildings",
        "satellite image of a helicopter on a runway in a large airport"
    ],
    "roundabout": [
        "overhead view of a circular traffic roundabout at an intersection",
        "aerial image of a road traffic circle in satellite imagery",
        "satellite view of a round-shaped road junction",
        "remote sensing image of a roundabout with multiple connecting roads",
        "aerial view of a traffic circle in an urban setting",
        "satellite photo showing a large road roundabout from above"
    ],
    "soccer ball field": [
        "overhead view of a rectangular soccer or football field with goalposts",
        "aerial image showing a football pitch with white markings",
        "satellite view of a soccer field in remote sensing imagery",
        "remote sensing photo of a green soccer field in an open stadium",
        "aerial view of a football field in a community sports park",
        "satellite image showing multiple soccer fields side by side"
    ],
    "plane": [
        "overhead view of an airplane or aircraft at an airport in satellite imagery",
        "aerial image of a fixed-wing aircraft on a runway",
        "satellite view of a commercial or private plane from above",
        "remote sensing image showing multiple planes parked at a terminal",
        "aerial photo of an aircraft taxiing on a runway",
        "satellite image of a jet aircraft in an airport apron area"
    ],
    "harbor": [
        "a high-resolution overhead view of a harbor with visible piers and docks",
        "satellite image showing a port area with cargo facilities",
        "aerial photo of a waterfront harbor with ships docked",
        "remote sensing image of a large commercial harbor from above",
        "overhead view of a marina with multiple boats and docking structures",
        "satellite image showing a port area with cranes and shipping containers"
    ]
}

background_description = {"background": [
    "overhead view of terrain, vegetation, or buildings without target objects",
    "aerial image of generic land cover not belonging to any labeled category",
    "satellite view of an area with no target object present",
    "remote sensing photo of natural or urban areas without labeled objects",
    "aerial view of mixed land cover without any target categories",
    "satellite image showing background area without objects of interest"
  ]
}
