from dotasim.env.channel import Action, Channel, lua_literal

TOK = "abc123"


def test_action_file_literal(tmp_path):
    ch = Channel(tmp_path)
    ch.write_action(TOK, 3, Action("attack", -600, -600, "npc_dota_creep_badguys_melee"))
    assert (tmp_path / f"actions_{TOK}_3.lua").read_text() == \
        'return {idx=3, kind="attack", x=-600, y=-600, name="npc_dota_creep_badguys_melee"}'
    ch.write_action(TOK, 4, Action())
    assert (tmp_path / f"actions_{TOK}_4.lua").read_text() == 'return {idx=4, kind="none", x=0, y=0, name=""}'


def test_config_literal(tmp_path):
    ch = Channel(tmp_path)
    ch.write_config(10, {5: "npc_dota_hero_puck", 0: "npc_dota_hero_nevermore"}, {5: "valve", 0: "controlled"})
    assert (tmp_path / "config.lua").read_text() == (
        'return {frames=10, heroes={[0]="npc_dota_hero_nevermore", [5]="npc_dota_hero_puck"}, '
        'roles={[0]="controlled", [5]="valve"}}'
    )
    ch.write_calib(TOK, -89.867, 10)
    assert (tmp_path / f"calib_{TOK}.lua").read_text() == "return {t=-89.867, n=10}"


def test_old_action_files_are_pruned(tmp_path):
    ch = Channel(tmp_path)
    for i in range(8):
        ch.write_action(TOK, i, Action("move", 1, 2))
    present = sorted(int(p.stem.rsplit("_", 1)[1]) for p in tmp_path.glob(f"actions_{TOK}_*.lua"))
    assert present == [3, 4, 5, 6, 7]


def test_clean_removes_run_files_only(tmp_path):
    (tmp_path / "actions_x_1.lua").write_text("")
    (tmp_path / "actions_y_2.lua").write_text("")
    (tmp_path / "console_0.log").write_text("")
    (tmp_path / "core.lua").write_text("return {}")
    assert Channel(tmp_path).clean() == 3
    assert (tmp_path / "core.lua").exists()


def test_lua_literal_types():
    assert lua_literal({"a": 1.5, "b": True, "c": [1, "x"]}) == '{a=1.5, b=true, c={1, "x"}}'
