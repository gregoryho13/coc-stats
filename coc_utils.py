import re
from bs4 import BeautifulSoup as bs4
import asyncio
import nest_asyncio

def get_player_info(player, limit=['achievements','load_game_data','builder_hall','builder_troops']):
    # Filter player information on getting only the attributes
    player_attributes = [attr for attr in dir(player) if re.search(r'^[^_]+',attr) and not re.search(r'(_cls)$|^(get_)',attr) and not attr in limit]
    player_data = [getattr(player,attr) for attr in player_attributes]

    player_info = {}
    for attr, data in zip(player_attributes, player_data):
        if re.match(r'.*troops$|siege_machines',attr):
            for html in data:
                troop = bs4(str(html),'html.parser').troop
                player_info[troop['name'].lower().replace(' ', '_').replace('.','')] = int(troop['level'])
        elif re.match(r'.*heroes$',attr):
            for html in data:
                hero = bs4(str(html),'html.parser').hero
                player_info[hero['name'].lower().replace(' ', '_').replace('.','')] = int(hero['level'])
        elif re.match(r'hero_pets$',attr):
            for html in data:
                pet = bs4(str(html),'html.parser').pet
                player_info[pet['name'].lower().replace(' ', '_').replace('.','')] = int(pet['level'])
        elif re.match(r'.*spells$',attr):
            for html in data:
                spell = bs4(str(html),'html.parser').spell
                player_info[spell['name'].lower().replace(' ', '_').replace('.','')] = int(spell['level'])
        elif attr == 'legend_statistics':
            if data is not None:
                stat = bs4(str(data),'html.parser').legendstatistics
                player_info['legend_trophies'] = int(stat['legend_trophies'])
        elif attr == 'labels':
            for idx, label in enumerate(data):
                player_info['label'+str(idx)] = str(label).lower().replace(' ','_')
        else:
            if isinstance(data, list):
                for idx, elm in enumerate(data):
                    data[idx] = str(elm)

            player_info[attr] = data if type(data) is int or type(data) is str or type(data) is list else str(data)
    return player_info

async def get_clan_info(clan, limit=['badge','members'], detailed=False):
    if limit is None:
        limit = ['badge','members']
    else:
        limit = limit + ['badge', 'members']

    # Filter clan information on getting only the attributes excluding attributes in the limit
    clan_attributes = [attr for attr in dir(clan) if re.search(r'^[^_]+',attr) and not re.search(r'(_cls)$|^(get_)',attr) and not attr in limit]
    clan_data = [getattr(clan,attr) for attr in clan_attributes]

    clan_info = {}
    for attr, data in zip(clan_attributes, clan_data):
        if isinstance(data, list):
            for idx, elm in enumerate(data):
                data[idx] = str(elm)
        clan_info[attr] = data if type(data) is int or type(data) is str or type(data) is list else str(data)
    
    if detailed:
        members = []
        async for player in clan.get_detailed_members():
            player_info = get_player_info(player)
            members.append(player_info)
        clan_info['members'] = members
        
    else:
        clan_info['members'] = [player.name+player.tag async for player in clan.get_detailed_members()]
    
    return clan_info