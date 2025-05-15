import pandas as pd


class SwingLevel:
    def __init__(
        self,
    ):
        self.supertrend_value = "st_fpvalue"
        self.supertrend_direction = "st_fpdirection"
        self.first_period = {
            "high_till_now": 0,
            "high_till_now_datetime": None,
            "low_till_now": 0,
            "low_till_now_datetime": None,
            "direction": 0,
            "value": 0,
            "flat_count": 0,
        }
        self.dict_start_up_move = {
            "datetime": 0,
            "support": 0,
            "resistance": 0,
            "is_up_move": 0,
            "is_up_trend": 0,
            "level": 0,
            "trigger_datetime": 0,
        }
        self.dict_start_down_move = {
            "datetime": 0,
            "support": 0,
            "resistance": 0,
            "is_down_move": 0,
            "is_up_move": 0,
            "is_up_trend": 0,
            "level": 0,
            "trigger_datetime": 0,
        }
        self.df_swing_levels = pd.DataFrame(
            columns=[
                "datetime",
                "is_up_trend",
                "is_up_move",
                "level",
                "trigger_datetime",
            ]
        )
        self.df_swing_levels.set_index("datetime", inplace=True)

    def set_period_values(self, candle, datetime_index):
        if (
            self.first_period["high_till_now"] == 0
            or self.first_period["high_till_now"] < candle["high"]
        ):
            self.first_period["high_till_now"] = candle["high"]
            self.first_period["high_till_now_datetime"] = datetime_index

        if (
            self.first_period["low_till_now"] == 0
            or self.first_period["low_till_now"] > candle["low"]
        ):
            self.first_period["low_till_now"] = candle["low"]
            self.first_period["low_till_now_datetime"] = datetime_index

        self.first_period["direction"] = candle[self.supertrend_direction]
        self.first_period["value"] = candle[self.supertrend_value]

    def create_new_start_of_upmove(self, trigger_datetime, is_up_move=True):
        new_start_of_upmove_row = {
            "datetime": 0,
            "support": 0,
            "resistance": 0,
            "is_up_move": 0,
            "is_up_trend": 0,
            "level": 0,
            "trigger_datetime": 0,
        }

        if is_up_move:
            new_start_of_upmove_row["datetime"] = self.first_period[
                "low_till_now_datetime"
            ]
            new_start_of_upmove_row["level"] = self.first_period["low_till_now"]
        else:
            new_start_of_upmove_row["datetime"] = self.first_period[
                "high_till_now_datetime"
            ]
            new_start_of_upmove_row["level"] = self.first_period["high_till_now"]

        new_start_of_upmove_row["support"] = self.first_period["low_till_now"]
        new_start_of_upmove_row["resistance"] = self.first_period["high_till_now"]
        new_start_of_upmove_row["is_up_move"] = is_up_move
        new_start_of_upmove_row["is_up_trend"] = True
        new_start_of_upmove_row["trigger_datetime"] = trigger_datetime

        return new_start_of_upmove_row

    def create_new_start_of_downmove(
        self,
        trigger_datetime,
        is_down_move=True,
    ):
        new_start_of_downmove_row = {
            "datetime": 0,
            "support": 0,
            "resistance": 0,
            "is_down_move": 0,
            "is_up_move": 0,
            "is_up_trend": 0,
            "level": 0,
            "trigger_datetime": 0,
        }

        if is_down_move:
            new_start_of_downmove_row["datetime"] = self.first_period[
                "high_till_now_datetime"
            ]
            new_start_of_downmove_row["level"] = self.first_period["high_till_now"]
            new_start_of_downmove_row["is_up_move"] = False
        else:
            new_start_of_downmove_row["datetime"] = self.first_period[
                "low_till_now_datetime"
            ]
            new_start_of_downmove_row["level"] = self.first_period["low_till_now"]
            new_start_of_downmove_row["is_up_move"] = True

        new_start_of_downmove_row["support"] = self.first_period["low_till_now"]
        new_start_of_downmove_row["resistance"] = self.first_period["high_till_now"]
        new_start_of_downmove_row["is_down_move"] = is_down_move
        new_start_of_downmove_row["is_up_trend"] = False
        new_start_of_downmove_row["trigger_datetime"] = trigger_datetime

        return new_start_of_downmove_row

    def initialize_period_dict(self, bar_datetime, candle):
        self.first_period = {
            "high_till_now": candle["high"],
            "high_till_now_datetime": bar_datetime,
            "low_till_now": candle["low"],
            "low_till_now_datetime": bar_datetime,
            "direction": candle[self.supertrend_direction],
            "value": candle[self.supertrend_value],
            "flat_count": 0,
        }

    def should_insert_trend_pullback_row(self, is_start_of_up_move=False):
        should_insert = False

        if is_start_of_up_move:
            # Since we cover a long duration we ignore swing points till we have a crossover in supertrend.
            if self.dict_start_up_move["level"] == 0:
                return should_insert

            if self.dict_start_up_move["is_up_move"]:
                should_insert = True
        else:
            if self.dict_start_up_move["level"] == 0:
                return should_insert

            if self.dict_start_down_move["is_down_move"]:
                should_insert = True

        return should_insert

    # This will check if close is higher than resistance of last record and there is a pullback
    def should_insert_trend_continuation_row(
        self, bar_high_or_low, is_start_of_up_move=False
    ):
        should_insert = False

        if is_start_of_up_move:
            # Since we cover a long duration we ignore swing points till we have a crossover in supertrend.
            if self.dict_start_up_move["level"] == 0:
                return should_insert

            if (
                not self.dict_start_up_move["is_up_move"]
                and bar_high_or_low > self.dict_start_up_move["resistance"]
            ):
                should_insert = True
        else:
            if self.dict_start_up_move["level"] == 0:
                return should_insert

            if (
                not self.dict_start_down_move["is_down_move"]
                and bar_high_or_low < self.dict_start_down_move["support"]
            ):
                should_insert = True

        return should_insert

    def loop_data_add_swing_levels(self, df_supertrend):

        for datetime_index, candle in df_supertrend.iterrows():
            self.loop_min_create_data_rows(datetime_index, candle)

        return self.df_swing_levels

    def loop_min_create_data_rows(self, datetime_index, candle):
        add_up_move = False
        add_down_move = False
        if self.first_period["direction"] == 0:
            self.initialize_period_dict(datetime_index, candle)

        # There is two possibilities first is crossover to down trend and second is crossover to up trend
        elif candle[self.supertrend_direction] != self.first_period["direction"]:
            # 1. We have to save to either start of up move or start of down move
            if candle[self.supertrend_direction] == 1:
                # We have crossover to an uptrend
                new_start_of_upmove_row = self.create_new_start_of_upmove(
                    datetime_index,
                    is_up_move=True,
                )
                add_up_move = True
                self.dict_start_up_move = new_start_of_upmove_row
            else:
                # We have crossover to an downtrend
                new_start_of_downmove_row = self.create_new_start_of_downmove(
                    datetime_index,
                    is_down_move=True,
                )
                add_down_move = True
                self.dict_start_down_move = new_start_of_downmove_row

            # reset the period to the current candle
            self.initialize_period_dict(datetime_index, candle)
        else:
            # This condition is to add pullback when first period super trend flattens
            if candle[self.supertrend_value] == self.first_period["value"]:
                if self.first_period["flat_count"] == 0:
                    self.set_period_values(candle, datetime_index)
                    self.first_period["flat_count"] = 1
                elif self.first_period["flat_count"] == 1:
                    # 1. If this is start of down move in a uptrend then we have to check if there is a prior record of start of down move.
                    # 2. If no prior start of down move We have to save to either start of up move or start of down move
                    if candle[self.supertrend_direction] == 1:
                        # This is a pullback in a up move
                        # If the last record has column is_up_move as True then only we insert a record
                        should_insert = self.should_insert_trend_pullback_row(
                            is_start_of_up_move=True
                        )

                        if should_insert:
                            new_start_of_upmove_row = self.create_new_start_of_upmove(
                                datetime_index,
                                is_up_move=False,
                            )
                            new_start_of_upmove_row["support"] = candle["low"]
                            add_up_move = True
                            self.dict_start_up_move = new_start_of_upmove_row
                    else:
                        # This is a pullback in a down move
                        should_insert = self.should_insert_trend_pullback_row(
                            is_start_of_up_move=False
                        )

                        if should_insert:
                            new_start_of_downmove_row = (
                                self.create_new_start_of_downmove(
                                    datetime_index,
                                    is_down_move=False,
                                )
                            )
                            new_start_of_downmove_row["resistance"] = candle["high"]
                            add_down_move = True
                            self.dict_start_down_move = new_start_of_downmove_row

                    self.initialize_period_dict(datetime_index, candle)
                    self.first_period["flat_count"] = 2
                else:
                    # We have already registered the pullback and now we either have crossover or continuation of trend
                    self.set_period_values(candle, datetime_index)
                    self.first_period["flat_count"] += 1
            else:
                # This part of code gets executed when price is in trend continuation.
                if candle[self.supertrend_direction] == 1:
                    # Since we cover a long duration we ignore swing points till we have a crossover.
                    # Verifying prior record is a pullback
                    should_insert_record = self.should_insert_trend_continuation_row(
                        candle["high"],
                        is_start_of_up_move=True,
                    )
                    if should_insert_record:
                        new_start_of_upmove_row = self.create_new_start_of_upmove(
                            datetime_index,
                            is_up_move=True,
                        )
                        add_up_move = True
                        self.dict_start_up_move = new_start_of_upmove_row

                        self.initialize_period_dict(datetime_index, candle)
                    else:
                        self.set_period_values(candle, datetime_index)
                else:
                    should_insert_record = self.should_insert_trend_continuation_row(
                        candle["low"],
                        is_start_of_up_move=False,
                    )
                    if should_insert_record:
                        new_start_of_downmove_row = self.create_new_start_of_downmove(
                            datetime_index,
                            is_down_move=True,
                        )
                        add_down_move = True
                        self.initialize_period_dict(datetime_index, candle)
                        self.dict_start_down_move = new_start_of_downmove_row
                    else:
                        self.set_period_values(candle, datetime_index)
                self.first_period["flat_count"] = 0

        # new_start_of_upmove_row, new_start_of_downmove_row
        if add_up_move == False and add_down_move == False:
            return

        if add_up_move == True:
            df = pd.DataFrame([new_start_of_upmove_row])
        elif add_down_move == True:
            df = pd.DataFrame([new_start_of_downmove_row])
        else:
            return

        df.set_index("datetime", inplace=True)
        columns_to_add = ["is_up_trend", "is_up_move", "level", "trigger_datetime"]
        df = df[columns_to_add]
        self.df_swing_levels = pd.concat(
            [self.df_swing_levels if not self.df_swing_levels.empty else None, df],
            ignore_index=True,
        )
        self.df_swing_levels.sort_index(inplace=True)
