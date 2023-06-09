#include "efa_unit_tests.h"

struct rxr_env orig_rxr_env = {0};

/* Runs once before all tests */
static int efa_unit_test_mocks_group_setup(void **state)
{
	struct efa_resource *resource;
	resource = calloc(1, sizeof(struct efa_resource));
	*state = resource;

	orig_rxr_env = rxr_env;

	return 0;
}

/* Runs once after all tests */
static int efa_unit_test_mocks_group_teardown(void **state)
{
	struct efa_resource *resource = *state;
	free(resource);

	return 0;
}

/* Runs before every test */
static int efa_unit_test_mocks_setup(void **state)
{
	/* Zero out *resource */
	struct efa_resource *resource = *state;
	memset(resource, 0, sizeof(struct efa_resource));

	return 0;
}

/* Runs after every test */
static int efa_unit_test_mocks_teardown(void **state)
{
	struct efa_resource *resource = *state;
	efa_unit_test_resource_destruct(resource);

	efa_mock_ibv_send_wr_list_destruct(&g_ibv_send_wr_list);

	g_efa_unit_test_mocks = (struct efa_unit_test_mocks) {
		.local_host_id = 0,
		.peer_host_id = 0,
		.ibv_create_ah = __real_ibv_create_ah,
		.efadv_query_device = __real_efadv_query_device,
#if HAVE_EFADV_CQ_EX
		.efadv_create_cq = __real_efadv_create_cq,
#endif
#if HAVE_NEURON
		.neuron_alloc = __real_neuron_alloc,
#endif
		.ofi_copy_from_hmem_iov = __real_ofi_copy_from_hmem_iov,
	};

	/* Reset environment */
	rxr_env = orig_rxr_env;
	unsetenv("FI_EFA_USE_DEVICE_RDMA");

	return 0;
}

int main(void)
{
	int ret;
	/* Requires an EFA device to work */
	const struct CMUnitTest efa_unit_tests[] = {
		cmocka_unit_test_setup_teardown(test_av_insert_duplicate_raw_addr, efa_unit_test_mocks_setup, efa_unit_test_mocks_teardown),
		cmocka_unit_test_setup_teardown(test_av_insert_duplicate_gid, efa_unit_test_mocks_setup, efa_unit_test_mocks_teardown),
		cmocka_unit_test_setup_teardown(test_efa_device_construct_error_handling, efa_unit_test_mocks_setup, efa_unit_test_mocks_teardown),
		cmocka_unit_test_setup_teardown(test_rxr_ep_ignore_missing_host_id_file, efa_unit_test_mocks_setup, efa_unit_test_mocks_teardown),
		cmocka_unit_test_setup_teardown(test_rxr_ep_has_valid_host_id, efa_unit_test_mocks_setup, efa_unit_test_mocks_teardown),
		cmocka_unit_test_setup_teardown(test_rxr_ep_ignore_short_host_id, efa_unit_test_mocks_setup, efa_unit_test_mocks_teardown),
		cmocka_unit_test_setup_teardown(test_rxr_ep_ignore_non_hex_host_id, efa_unit_test_mocks_setup, efa_unit_test_mocks_teardown),
		cmocka_unit_test_setup_teardown(test_rxr_ep_handshake_receive_and_send_valid_host_ids_with_connid, efa_unit_test_mocks_setup, efa_unit_test_mocks_teardown),
		cmocka_unit_test_setup_teardown(test_rxr_ep_handshake_receive_and_send_valid_host_ids_without_connid, efa_unit_test_mocks_setup, efa_unit_test_mocks_teardown),
		cmocka_unit_test_setup_teardown(test_rxr_ep_handshake_receive_valid_peer_host_id_and_do_not_send_local_host_id, efa_unit_test_mocks_setup, efa_unit_test_mocks_teardown),
		cmocka_unit_test_setup_teardown(test_rxr_ep_handshake_receive_without_peer_host_id_and_do_not_send_local_host_id, efa_unit_test_mocks_setup, efa_unit_test_mocks_teardown),
		cmocka_unit_test_setup_teardown(test_rxr_ep_cq_create_error_handling, efa_unit_test_mocks_setup, efa_unit_test_mocks_teardown),
		cmocka_unit_test_setup_teardown(test_rxr_ep_pkt_pool_flags, efa_unit_test_mocks_setup, efa_unit_test_mocks_teardown),
		cmocka_unit_test_setup_teardown(test_rxr_ep_pkt_pool_page_alignment, efa_unit_test_mocks_setup, efa_unit_test_mocks_teardown),
		cmocka_unit_test_setup_teardown(test_rxr_ep_dc_atomic_error_handling, efa_unit_test_mocks_setup, efa_unit_test_mocks_teardown),
		cmocka_unit_test_setup_teardown(test_rxr_ep_send_with_shm_no_copy, efa_unit_test_mocks_setup, efa_unit_test_mocks_teardown),
		cmocka_unit_test_setup_teardown(test_dgram_cq_read_empty_cq, efa_unit_test_mocks_setup, efa_unit_test_mocks_teardown),
		cmocka_unit_test_setup_teardown(test_dgram_cq_read_bad_wc_status_unresponsive_receiver, efa_unit_test_mocks_setup, efa_unit_test_mocks_teardown),
		cmocka_unit_test_setup_teardown(test_ibv_cq_ex_read_empty_cq, efa_unit_test_mocks_setup, efa_unit_test_mocks_teardown),
		cmocka_unit_test_setup_teardown(test_ibv_cq_ex_read_failed_poll, efa_unit_test_mocks_setup, efa_unit_test_mocks_teardown),
		cmocka_unit_test_setup_teardown(test_ibv_cq_ex_read_bad_send_status_unresponsive_receiver, efa_unit_test_mocks_setup, efa_unit_test_mocks_teardown),
		cmocka_unit_test_setup_teardown(test_ibv_cq_ex_read_bad_send_status_unresponsive_receiver_missing_peer_host_id, efa_unit_test_mocks_setup, efa_unit_test_mocks_teardown),
		cmocka_unit_test_setup_teardown(test_ibv_cq_ex_read_bad_send_status_invalid_qpn, efa_unit_test_mocks_setup, efa_unit_test_mocks_teardown),
		cmocka_unit_test_setup_teardown(test_ibv_cq_ex_read_bad_send_status_message_too_long, efa_unit_test_mocks_setup, efa_unit_test_mocks_teardown),
		cmocka_unit_test_setup_teardown(test_ibv_cq_ex_read_bad_recv_status, efa_unit_test_mocks_setup, efa_unit_test_mocks_teardown),
		cmocka_unit_test_setup_teardown(test_ibv_cq_ex_read_recover_forgotten_peer_ah, efa_unit_test_mocks_setup, efa_unit_test_mocks_teardown),
		cmocka_unit_test_setup_teardown(test_ibv_cq_ex_read_ignore_removed_peer, efa_unit_test_mocks_setup, efa_unit_test_mocks_teardown),
		cmocka_unit_test_setup_teardown(test_rdm_fallback_to_ibv_create_cq_ex_cq_read_ignore_forgotton_peer, efa_unit_test_mocks_setup, efa_unit_test_mocks_teardown),
		cmocka_unit_test_setup_teardown(test_info_open_ep_with_wrong_info, efa_unit_test_mocks_setup, efa_unit_test_mocks_teardown),
		cmocka_unit_test_setup_teardown(test_info_open_ep_with_api_1_1_info, efa_unit_test_mocks_setup, efa_unit_test_mocks_teardown),
		cmocka_unit_test_setup_teardown(test_info_check_shm_info, efa_unit_test_mocks_setup, efa_unit_test_mocks_teardown),
		cmocka_unit_test_setup_teardown(test_info_check_hmem_cuda_support_on_api_lt_1_18, NULL, NULL),
		cmocka_unit_test_setup_teardown(test_info_check_hmem_cuda_support_on_api_ge_1_18, NULL, NULL),
		cmocka_unit_test_setup_teardown(test_info_check_no_hmem_support_when_not_requested, NULL, NULL),
		cmocka_unit_test_setup_teardown(test_efa_use_device_rdma_env1_opt1, efa_unit_test_mocks_setup, efa_unit_test_mocks_teardown),
		cmocka_unit_test_setup_teardown(test_efa_use_device_rdma_env0_opt0, efa_unit_test_mocks_setup, efa_unit_test_mocks_teardown),
		cmocka_unit_test_setup_teardown(test_efa_use_device_rdma_env1_opt0, efa_unit_test_mocks_setup, efa_unit_test_mocks_teardown),
		cmocka_unit_test_setup_teardown(test_efa_use_device_rdma_env0_opt1, efa_unit_test_mocks_setup, efa_unit_test_mocks_teardown),
		cmocka_unit_test_setup_teardown(test_efa_use_device_rdma_env1, efa_unit_test_mocks_setup, efa_unit_test_mocks_teardown),
		cmocka_unit_test_setup_teardown(test_efa_use_device_rdma_env0, efa_unit_test_mocks_setup, efa_unit_test_mocks_teardown),
		cmocka_unit_test_setup_teardown(test_efa_use_device_rdma_opt1, efa_unit_test_mocks_setup, efa_unit_test_mocks_teardown),
		cmocka_unit_test_setup_teardown(test_efa_use_device_rdma_opt0, efa_unit_test_mocks_setup, efa_unit_test_mocks_teardown),
		cmocka_unit_test_setup_teardown(test_efa_use_device_rdma_opt_old, efa_unit_test_mocks_setup, efa_unit_test_mocks_teardown),
		cmocka_unit_test_setup_teardown(test_efa_hmem_info_update_neuron, efa_unit_test_mocks_setup, efa_unit_test_mocks_teardown),
	};

	cmocka_set_message_output(CM_OUTPUT_XML);

	ret = cmocka_run_group_tests_name("efa unit tests", efa_unit_tests, efa_unit_test_mocks_group_setup, efa_unit_test_mocks_group_teardown);

	return ret;
}
