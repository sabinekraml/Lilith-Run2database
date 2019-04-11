##########################################################################
#
#  This file is part of Lilith
#  made by J. Bernon and B. Dumont
#  extended by TRAN Quang Loc (TQL) and LE Duc Ninh (LDN), March 2019
#
#  Web page: http://lpsc.in2p3.fr/projects-th/lilith/
#
#  In case of questions email sabine.kraml@lpsc.in2p3.fr
#
#
#    Lilith is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    Lilith is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with Lilith.  If not, see <http://www.gnu.org/licenses/>.
#
##########################################################################

from ..errors import LikelihoodComputationError
import numpy as np

def compute_likelihood(exp_mu, user_mu):
    """Computes the likelihood from experimental mu and user mu."""
    likelihood_results = []
    l = 0. # actually -2log(likelihood)
    for mu in exp_mu:
        # compute user mu value scaled to efficiencies
        user_mu_effscaled = {}
        try:
            user_mu_effscaled["x"] = 0.
            for (prod,decay),eff_prod in mu["eff"]["x"].items():
		if mu["sqrts"] not in ["1.96","7","8","7.","8.","7.0","8.0","7+8"] and (prod == "ggH" or prod == "VBF"):
		     prod == prod + "13"
                user_mu_effscaled["x"] += eff_prod*user_mu[prod,decay]

            if mu["dim"] == 2:
                user_mu_effscaled["y"] = 0.
                for (prod,decay),eff_prod in mu["eff"]["y"].items():
		    if mu["sqrts"] not in ["1.96","7","8","7.","7.0","8.0","8.","7+8"] and (prod == "ggH" or prod == "VBF"):
                         prod == prod + "13"
                    user_mu_effscaled["y"] += eff_prod*user_mu[prod,decay]
        except KeyError as s:
            if "s" in ["eff", "x", "y"]:
                # the experimental mu dictionnary is not filled correctly
                raise LikelihoodComputationError(
                    'there are missing elements in exp_mu: key "' + str(s) +
                    '" is not found')
            else:
                # the total user mu dictionnary is not filled correctly
                raise LikelihoodComputationError(
                    'there are missing elements in user_mu_tot: key "' +
                    str(s) + '" is not found')

        try:
            # likelihood computation in case of a type="normal"
            if mu["type"] == "n":
                if mu["dim"] == 1:
            # likelihood computation in Odinary Gaussian approximation
                    if user_mu_effscaled["x"] < mu["bestfit"]["x"]:
                        unc = mu["param"]["uncertainty"]["left"]
                    else:
                        unc = mu["param"]["uncertainty"]["right"]
                    cur_l = ((mu["bestfit"]["x"] - user_mu_effscaled["x"])**2/unc**2)

                elif mu["dim"] == 2:
            # likelihood computation in Odinary Gaussian approximation
                    a = mu["param"]["a"]
                    b = mu["param"]["b"]
                    c = mu["param"]["c"]

                    cur_l = a*(mu["bestfit"]["x"] - user_mu_effscaled["x"])**2
                    cur_l += c*(mu["bestfit"]["y"] - user_mu_effscaled["y"])**2
                    cur_l += (2*b*(mu["bestfit"]["x"] - user_mu_effscaled["x"])
                             * (mu["bestfit"]["y"] - user_mu_effscaled["y"]))

            # likelihood computation in case of a type="variable normal"
            if mu["type"] == "vn":
                if mu["dim"] == 1:
            # likelihood computation in Variable Gaussian 2, see Barlow arXiv:physics/0406120v1, Eq. 18
                    unc_left = abs(mu["param"]["uncertainty"]["left"])
                    unc_right = mu["param"]["uncertainty"]["right"]
                    num = user_mu_effscaled["x"] - mu["bestfit"]["x"]
                    den = unc_left*unc_right + (unc_right - unc_left)*num
                    if den == 0:
                        cur_l = 0.
                    else:
                        cur_l = num**2/den
                if mu["dim"] == 2:
            # likelihood computation in Variable Gaussian 2, see Barlow arXiv:physics/0406120v1, Eq. 18
                    p = mu["param"]["correlation"]
                    sig1p = mu["param"]["uncertainty"]["x"]["right"]
                    sig1m = abs(mu["param"]["uncertainty"]["x"]["left"])
                    sig2p = mu["param"]["uncertainty"]["y"]["right"]
                    sig2m = abs(mu["param"]["uncertainty"]["y"]["left"])
                    z10 = mu["bestfit"]["x"]
                    z20 = mu["bestfit"]["y"]
                    z1 = user_mu_effscaled["x"]
                    z2 = user_mu_effscaled["y"]
                    V1 = sig1p*sig1m
                    V1e = sig1p - sig1m
                    V2 = sig2p*sig2m
                    V2e = sig2p - sig2m
                    V1f = V1 + V1e*(z1-z10)
                    V2f = V2 + V2e*(z2 - z20)
                    cur_l = 1.0/(1-p**2)*((z1-z10)**2/V1f-2*p*(z1-z10)*(z2-z20)/np.sqrt(V1f*V2f)+(z2-z20)**2/V2f)

            # likelihood computation in case of a type="Poisson"
            if mu["type"] == "p":
                if mu["dim"] == 1:
            # likelihood computation in Poisson, see Barlow arXiv:physics/0406120v1, Eq. 10a
                    alpha = mu["param"]["alpha"]
                    nu = mu["param"]["nu"]
                    cen2 = mu["bestfit"]["x"]
                    if 1 + alpha * (user_mu_effscaled["x"] - cen2) / nu > 0:
                        cur_l = -2 * (-alpha * (user_mu_effscaled["x"] - cen2) + nu * np.log(1 + alpha * (user_mu_effscaled["x"] - cen2) / nu))
                    else:
                        cur_l = 0.

            # likelihood computation in case of a type="full" (exact likelihood provided in terms of a grid file)
            if mu["type"] == "f":
                if mu["dim"] == 1:
                    cur_l = mu["Lxy"](user_mu_effscaled["x"]) - mu["LChi2min"]
                elif mu["dim"] == 2:
                    cur_l = (mu["Lxy"](user_mu_effscaled["x"],
                                       user_mu_effscaled["y"])[0][0]
                            - mu["LChi2min"])
        except KeyError as s:
            raise LikelihoodComputationError(
                'there are missing elements in exp_mu: key "' + s +
                '" is not found')

        l += cur_l
        likelihood_results.append(
            {"experiment": mu["experiment"], "source": mu["source"],
            "sqrts": mu["sqrts"], "dim": mu["dim"],
            "type": mu["type"], "eff": mu["eff"], "l": cur_l})

    return likelihood_results, l
