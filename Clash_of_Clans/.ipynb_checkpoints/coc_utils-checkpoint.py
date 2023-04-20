import re
from bs4 import BeautifulSoup as bs4
import asyncio
import nest_asyncio
from IPython.display import clear_output

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

async def get_clan_info(client, clan_tag, limit=None, detailed=False, omit_members=False):
    clan = await client.get_clan(clan_tag)
    
    # Exclude badge information and members (member data is implemented elsewhere)
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
    
    # If detailed, collect all clan-war-useful data from each member
    if detailed:
        members = []
        member_attack_wins = []
        best_trophies = []
        donations = []
        
        pets = ['lassi','electro_owl','mighty_yak','unicorn']
        pet_lvls = {'lassi': [], 'electro_owl': [], 'mighty_yak': [], 'unicorn': []}
        
        heroes = ['barbarian_king','archer_queen','grand_warden','royal_champion']
        hero_lvls = {'barbarian_king': [], 'archer_queen': [], 'grand_warden': [], 'royal_champion': []}
        
        legend_trophies = []
        town_hall = []
        trophies = []
        war_stars = []
        total_war_opted_in = 0
        async for player in clan.get_detailed_members():
            player_info = await get_player_info(client, player.tag)
            members.append(player.name+player.tag)
            
            member_attack_wins.append(player_info['attack_wins']) if player_info['attack_wins'] is not str else member_attack_wins.append(0)
            best_trophies.append(player_info['best_trophies']) if player_info['best_trophies'] is not str else best_trophies.append(0)
            donations.append(player_info['donations']) if player_info['donations'] is not str else donations.append(0)
            
            for pet in pets:
                pet_lvls[pet].append(player_info[pet]) if pet in player_info else pet_lvls[pet].append(0)
            for hero in heroes:
                hero_lvls[hero].append(player_info[hero]) if hero in player_info else hero_lvls[hero].append(0)
            
            
            legend_trophies.append(player_info['legend_trophies']) if 'legend_trophies' in player_info else legend_trophies.append(0)
            
            town_hall.append(player_info['town_hall'])
            trophies.append(player_info['trophies'])
            war_stars.append(player_info['war_stars'])
            
            total_war_opted_in += 1 if player_info['war_opted_in']=='True' else 0
        
        # Add sorted values list to clan_info dictionary
        clan_info['member_attack_wins'] = sorted(member_attack_wins, reverse=True)
        clan_info['best_trophies'] = sorted(best_trophies, reverse=True)
        clan_info['donations'] = sorted(donations, reverse=True)
        
        for pet in pets:
            clan_info[pet] = sorted(pet_lvls[pet], reverse=True)
        for hero in heroes:
            clan_info[hero] = sorted(hero_lvls[hero], reverse=True)
        
        clan_info['legend_trophies'] = sorted(legend_trophies, reverse=True)
        clan_info['town_hall'] = sorted(town_hall, reverse=True)
        clan_info['trophies'] = sorted(trophies, reverse=True)
        clan_info['war_stars'] = sorted(war_stars, reverse=True)
        clan_info['total_war_opted_in'] = total_war_opted_in
        if not omit_members:
            clan_info['members'] = members
        
    else:
        if not omit_members:
            members = []
            async for member in clan.get_detailed_members():
                members.append(member.name+member.tag)
            clan_info['members'] = members
    
    return clan_info

async def get_warlog_info(client, clan_tag, clan_details=False):
    war_log = await client.get_warlog(clan_tag)
    
    war_log_info = []
    for i, war in enumerate(war_log):
        if not war_log[i].is_league_entry:
            clear_output(wait=True)
            print(f'Logging information on war against {war_log[i].opponent.name}{war_log[i].opponent.tag}...')
            print(f'War {i+1} of {len(war_log)}')
            warlog_attributes = [attr for attr in dir(war) if re.search(r'^[^_]+',attr) and not re.search(r'(_cls)$|^(get_)',attr) and not attr in ['attacks_per_member','is_league_entry']]
            warlog_info = [getattr(war,attr) for attr in warlog_attributes]

            war_info = {}
            for attr, info in zip(warlog_attributes, warlog_info):
                if attr == 'clan' or attr == 'opponent':
                    limit = ['badge','members','attacks','defenses','is_opponent']
                    if attr == 'opponent':
                        limit += ['exp_earned','attacks_used','average_attack_duration']

                    clan_war_attributes = [attr for attr in dir(info) if re.search(r'^[^_]+',attr) and not re.search(r'(_cls)$|^(get_)',attr) and not attr in limit]
                    for war_attr in clan_war_attributes:
                        war_info[attr +'_'+war_attr] = getattr(info, war_attr)
                elif attr == 'end_time':
                    war_info[attr] = format(info.time)
                else:
                    war_info[attr] = info

            if clan_details:
                details = await get_clan_info(client, war_info['opponent_tag'], limit=['tag','name','share_link'],detailed=True, omit_members=True)
                war_info = war_info | details
                war_info_del = ['clan_name','clan_share_link','clan_tag','clan_average_attack_duration','labels']
                for tag in war_info_del:
                    del war_info[tag]

            war_log_info.append(war_info)
    clear_output()
    print(f'Wars {i+1} of {len(war_log)} finished logging')
    return war_log_info