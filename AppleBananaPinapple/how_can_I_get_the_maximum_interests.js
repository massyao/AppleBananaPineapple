function calc_my_interests(init_saving) {
	const total = 70
	const interest_rate = 0.045 / 12
	// const init_saving = 0
	let saving = init_saving
	const time = (total - init_saving) / 2
	let interests  = 0
	for (i = 0; i < time ; i++) {
		saving = saving + 2
		let interests_one_month = saving * interest_rate
		interests = interests + interests_one_month
	}
	const low_rate_interests = 70 * 0.02 / 12 * (35 - time)
	return (interests + low_rate_interests).toFixed(2) - 0
}

(() => {
	const total = 70
	const result = Array.from(new Array(70), (e, i) => i).map((init_saving_iter) => calc_my_interests(init_saving_iter))
	const the_max_interest = Math.max(...result)
	const init_saving = result.indexOf(the_max_interest)
	console.log('you should put ' , init_saving, ' at first month', ' so you can get the_max_interest ', the_max_interest )
	console.log('in contrast if you put all the money into saving account, you will get', 70 * 0.02 / 12 * 35)
})()
