from mjai import Bot
from mjai.bot.consts import MJAI_VEC34_TILES
import numpy as np
import sys
import json

class WeightsBot(Bot):
    def __init__(self, weights, player_id: int=0):
        super().__init__(player_id)
        self.action_weights = np.array(weights)
        #discard, riichi, tsumo_agari, ron_agari, ryukyoku,
        #kakan, daiminkan, ankan, pon, chi, pass
        self.actions = [
            lambda: self.rule_base_discard(),
            lambda: self.action_riichi(),
            lambda: self.action_tsumo_agari(),
            lambda: self.action_ron_agari(),
            lambda: self.action_ryukyoku(),
            lambda: self.rule_base_kakan(),
            lambda: self.rule_base_daiminkan(),
            lambda: self.rule_base_ankan(),
            lambda: self.rule_base_pon(),
            lambda: self.rule_base_chi()
            ]
    
    @property
    def discardable_tiles(self) -> list[str]: #Overwriting base class method because it's wrong
        discardable_tiles = list(
            set(
                [
                    tile
                    for tile in self.tehai_mjai
                    if not self.forbidden_tiles[tile[:2]]
                ]
            )
        )
        return discardable_tiles
    
    def possible_action_arr(self):
        possible_actions = []
        if self.can_discard:
            possible_actions.append(0)
        if self.can_riichi:
            possible_actions.append(1)
        if self.can_tsumo_agari:
            possible_actions.append(2)
        if self.can_ron_agari:
            possible_actions.append(3)
        if self.can_ryukyoku:
            possible_actions.append(4)
        if self.can_kakan:
            possible_actions.append(5)
        if self.can_daiminkan:
            possible_actions.append(6)
        if self.can_ankan:
            possible_actions.append(7)
        if self.can_pon:
            possible_actions.append(8)
        if self.can_chi:
            possible_actions.append(9)
        return np.array(possible_actions)
    
    def rule_base_discard(self):
        if self.self_riichi_accepted:
            return self.action_discard(self.last_self_tsumo)

        candidates = self.find_improving_tiles()
        for candidate in candidates:
            discard_tile = candidate["discard_tile"]
            if self.forbidden_tiles.get(discard_tile[:2], True):
                continue
            return self.action_discard(discard_tile)

        return self.action_discard(
            self.last_self_tsumo or self.tehai_mjai[0]
        )
    
    def rule_base_kakan(self):
        events = self.get_call_events(self.player_id)
        for ev in events:
            if ev["type"] == "pon":
                pai = ev["pai"][:2]
                if self.tehai_vec34[MJAI_VEC34_TILES.index(pai)]:
                    if pai[0] == 5 and all([len(x)==2 for x in [ev["pai"]]+ev["consumed"]]):
                        return self.action_kakan(pai+"r")
                    else:
                        return self.action_kakan(pai)
        #This part should never be called
        return self.action_nothing()

    def rule_base_daiminkan(self):
        pai = self.last_kawa_tile[:2]
        consumed = []
        for tile in self.tehai_mjai:
            if pai in tile:
                consumed.append(tile)
        assert len(consumed) == 3
        return self.action_daiminkan(consumed)

    def rule_base_ankan(self):
        for idx in range(34):
            if self.tehai_vec34[idx] == 4:
                consumed = [MJAI_VEC34_TILES[idx] for _ in range(4)]
                if MJAI_VEC34_TILES[idx][0] == 5:
                    consumed[0] = consumed[0] + "r"
                return self.action_ankan(consumed)
        #This part should never be called
        return self.action_nothing()

    def rule_base_pon(self):
        pons = self.find_pon_candidates()
        for pon in pons:
            if pon["current_shanten"] > pon["next_shanten"]:
                return self.action_pon(consumed=pon["consumed"])
        return self.action_pon(consumed=pons[0]["consumed"])

    def rule_base_chi(self):
        chis = self.find_chi_candidates()
        best_ukeire = max([chi["next_ukeire"] for chi in chis])
        for chi in chis:
            if (
                chi["current_shanten"] > chi["next_shanten"]
                and chi["next_ukeire"] == best_ukeire
            ):
                return self.action_chi(consumed=chi["consumed"])
        return self.action_chi(consumed=chis[0]["consumed"])

    def think(self) -> str:
        if self.can_act:
            possible_actions = self.possible_action_arr()
            choice = possible_actions[np.argmax(self.action_weights[possible_actions])]
            return self.actions[choice]()
        else:
            return self.action_nothing()
    
    def react(self, input_str: str) -> str:
        try:
            events = json.loads(input_str)
            if len(events) == 0:
                raise ValueError("Empty events")
            for event in events:
                if event["type"] == "start_kyoku":
                    self.__discard_events = []
                    self.__call_events = []
                    self.__dora_indicators = []
                if event["type"] == "dora":
                    self.__dora_indicators.append(event["dora_marker"])
                if event["type"] == "dahai":
                    self.__discard_events.append(event)
                if event["type"] in [
                    "chi",
                    "pon",
                    "daiminkan",
                    "kakan",
                    "ankan",
                ]:
                    self.__call_events.append(event)

                self.action_candidate = self.player_state.update(
                    json.dumps(event)
                )

            # NOTE: Skip `think()` if the player's riichi is accepted and
            # no call actions are allowed.
            if (
                self.self_riichi_accepted
                and not (self.can_agari or self.can_kakan or self.can_ankan)
                and self.can_discard
            ):
                return self.action_discard(self.last_self_tsumo)

            resp = self.think()
            return resp

        except Exception as e:
            print(
                "===========================================", file=sys.stderr
            )
            print(f"Exception: {str(e)}", file=sys.stderr)
            print("Brief info:", file=sys.stderr)
            print(self.brief_info(), file=sys.stderr)
            print("", file=sys.stderr)

        return json.dumps({"type": "none"}, separators=(",", ":"))


WeightsBot([0.87709162,0.65932064,0.62645731,0.79081549,0.93703185,0.64617442,
 1.32771267,0.56352131,0.77811518,0.61813046], player_id=int(sys.argv[1])).start()