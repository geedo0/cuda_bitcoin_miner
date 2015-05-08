/*
 * EC527 Final Project
 * May 8, 2015
 * Gerardo Ravago - gerardo@gcr.me
 *
 * GPU Bitcoin Miner
 * Description
 * 
 * Special Thanks
 *  Luke Dashjr - libblkmaker and the accompanying example source
 *  Brad Conte - Reference implementation of SHA-256
 */

#include <assert.h>
#include <inttypes.h>
#include <stdint.h>
#include <arpa/inet.h>

#include <libbase58.h>
#include <blkmaker.h>
#include <blkmaker_jansson.h>

#include "utils.h"
#include "sha256.h"

#include "testinput.c"

#define PRINT_STATUS

//Use this to submit a request to the JSON RPC
//TODO: Submit to bitcoin-cli
static void send_json(json_t *req) {
	char *s = json_dumps(req, JSON_INDENT(2));
	puts(s);
	free(s);
}

int main(int argc, char**argv) {
	blktemplate_t *tmpl;
	json_t *req;
	json_error_t jsone;
	const char *err;
	
	//Provide a reference SHA-256 implementation for b58 check
	blkmk_sha256_impl = my_sha256;
	
	//Create the getblocktemplate request
	tmpl = blktmpl_create();
	assert(tmpl);
	req = blktmpl_request_jansson(blktmpl_addcaps(tmpl), NULL);
	assert(req);
	
	//Send request to bitcoind and receive work
	//TODO: Setup coinbase and bitcoind stuff
	send_json(req);
	json_decref(req);
	{
		//Fake the input data
		//TODO: Implement receive request
		req = json_loads(blkmaker_test_input, 0, &jsone);
		send_json(req);
	}
	assert(req);
	
	//Update block template with request results
	err = blktmpl_add_jansson(tmpl, req, time(NULL));
	json_decref(req);
	if (err)
	{
		fprintf(stderr, "Error adding block template: %s", err);
		assert(0 && "Error adding block template");
	}

	//Main mining loop
	//Check for expiration and work left
	//TODO: Infinite loop mode, when the block expires request a new block template
	while (blkmk_time_left(tmpl, time(NULL)) && blkmk_work_left(tmpl))
	{
		unsigned char data[80], hash[32];
		size_t datasz;
		unsigned int dataid;
		uint32_t nonce;
		
		//Get new block header
		datasz = blkmk_get_data(tmpl, data, sizeof(data), time(NULL), NULL, &dataid);
		assert(datasz >= 76 && datasz <= sizeof(data));
		
		verify_hash(my_sha256, data);

		//Nonce Mining Loop
		tick();
		for (nonce = 0; nonce < 0xffffffff; ++nonce)
		{
			*(uint32_t*)(&data[76]) = nonce;
			my_sha256(hash, data, 80);
			my_sha256(hash, hash, 32);
			//TODO: Properly check difficulty
			/*
			if (!*(uint32_t*)(&hash[28]))
				break;
				*/
			#ifdef PRINT_STATUS
			if (!(nonce % 0x1000))
			{
				printf("0x%8" PRIx32 " hashes done...\r", nonce);
				fflush(stdout);
			}
			#endif
		}
		tock();
		print_execution_time();
		printf("Found nonce: 0x%8" PRIx32 " \n", nonce);
		//Convert endianess so the block gets formed correctly
		nonce = ntohl(nonce);
		
		//Submit block to bitcoind
		req = blkmk_submit_jansson(tmpl, data, dataid, nonce);
		assert(req);
		send_json(req);
		break;
	}
	blktmpl_free(tmpl);
}
