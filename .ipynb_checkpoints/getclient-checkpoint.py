import coc
import asyncio



loop = asyncio.get_event_loop()

asyncio.run_coroutine_threadsafe(coc.login('limit.digit@gmail.com', 'cmsc3200201'), loop)

asyncio.run_coroutine_threadsafe(main(), loop)


async def main():
    player = await client.get_player("tag")
    print("{0.name} has {0.trophies} trophies!".format(player))

    clans = await client.search_clans(name="Best Clan Ever", limit=5)
    for clan in clans:
        print("{0.name} ({0.tag}) has {0.member_count} members".format(clan))

    try:
        war = await client.get_current_war("#clantag")
        print("{0.clan_tag} is currently in {0.state} state.".format(war))
    except coc.PrivateWarLog:
        print("Uh oh, they have a private war log!")