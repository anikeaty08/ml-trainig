from backend.utils.setup_manager import initialize_setup, install_packs, remove_pack, setup_status


def test_setup_initialization_and_pack_install():
    state = initialize_setup(
        profile_id="core",
        pack_ids=["core-tabular", "boosting-pack"],
        provider_ids=["ollama", "openai"],
        download_now=False,
    )

    assert state["setup_complete"] is True
    assert "core-tabular" in state["selected_pack_ids"]
    assert "boosting-pack" in state["deferred_pack_ids"]

    installed = install_packs(["boosting-pack"])
    assert "boosting-pack" in installed

    status = setup_status()
    assert any(pack["id"] == "boosting-pack" and pack["installed"] for pack in status["packs"])


def test_remove_pack_updates_state():
    initialize_setup(
        profile_id="core",
        pack_ids=["core-tabular"],
        provider_ids=["ollama"],
        download_now=True,
    )

    state = remove_pack("core-tabular")

    assert "core-tabular" not in state["installed_pack_ids"]
