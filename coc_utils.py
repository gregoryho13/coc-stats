import re
from bs4 import BeautifulSoup as bs4
import asyncio
import nest_asyncio

async def get_player_info(client, player_tag, limit=['achievements','load_game_data','builder_hall','builder_troops']):
    player = await client.get_player(player_tag)
    
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

async def get_clan_info(client, clan_tag, limit=['badge','members'], detailed=False):
    clan = await client.get_clan(clan_tag)
    
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
            player_info = get_player_info(client, player.tag)
            members.append(player_info)
        clan_info['members'] = members
        
    else:
        clan_info['members'] = [player.name+player.tag async for player in clan.get_detailed_members()]
    
    return clan_info

async def get_warlog_info(client, clan_tag):
    war_log = await client.get_warlog(clan_tag)
    
    war_log_info = []
    for i, war in enumerate(war_log):
        warlog_attributes = [attr for attr in dir(war) if re.search(r'^[^_]+',attr) and not re.search(r'(_cls)$|^(get_)',attr)]
        warlog_info = [getattr(war,attr) for attr in warlog_attributes]
        
        war_info = {}
        for attr, info in zip(warlog_attributes, warlog_info):
            if attr == 'clan' or attr == 'opponent':
                limit = ['badge','members','attacks','defenses','is_opponent']
                if attr == 'opponent':
                    limit += ['exp_earned','attacks_used']

                clan_war_attributes = [attr for attr in dir(info) if re.search(r'^[^_]+',attr) and not re.search(r'(_cls)$|^(get_)',attr) and not attr in limit]
                for war_attr in clan_war_attributes:
                    war_info[attr +'_'+war_attr] = getattr(info, war_attr)
            elif attr == 'end_time':
                war_info[attr] = format(info.time)
            else:
                war_info[attr] = info
        war_log_info.append(war_info)
    
    return war_log_info