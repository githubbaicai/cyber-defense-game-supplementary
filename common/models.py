

import numpy as np

from common.model_options import resolve_model_options


# ============================================

# ============================================

class Vulnerability:
    def __init__(self, name, base_success, base_attack_time,
                 attack_cost, attack_benefit, defend_loss, defense_success_prob,
                 defend_loss_destruction=0.0, weight_dld=0.0):
        self.name = name
        self.base_success = base_success
        self.base_attack_time = base_attack_time
        self.attack_cost = attack_cost
        self.attack_benefit = attack_benefit
        self.defend_loss = abs(float(defend_loss))
        self.defense_success_prob = defense_success_prob
        self.defend_loss_destruction = abs(float(defend_loss_destruction))
        self.weight_dld = weight_dld
        self.response_operational_disruption_cost = 0.0


class AttackStrategy:
    def __init__(self, target_component, vulnerability, current_path, prev_time,
                 ids_config, firewall_config, patch_config, *, model_options=None):
        self.model_options = resolve_model_options(model_options)
        self.target_component = target_component
        self.vulnerability = vulnerability
        self.current_path = current_path
        self.path_str = " → ".join(current_path + [target_component])
        self.prev_time = prev_time
        self.ids_config = ids_config
        self.firewall_config = firewall_config
        self.patch_config = patch_config

        self.base_attack_time = vulnerability.base_attack_time
        self.total_attack_time = self.base_attack_time + self.prev_time

        self.detected_prob = self.compute_detected_probability()

        self.passive_success_prob = (
            self.compute_passive_success_probability()
        )
        self.active_success_prob = (
            self.passive_success_prob
            * (1.0 - self.response_success_prob)
        )
        self.response_operational_disruption_cost = (
            self.vulnerability.defend_loss_destruction
            * self.vulnerability.weight_dld
            * self.response_success_prob
            * self.model_options.response_penalty_ratio
        )
        # Compatibility snapshot for no response. Use the explicit policy-aware
        # method when reporting a solved game's actual success probability.
        self.success_prob = self.compute_success_probability()

        self.attack_cost = vulnerability.attack_cost
        self.attack_benefit = vulnerability.attack_benefit
        self.defend_loss = vulnerability.defend_loss

    def compute_detected_probability(self):
        ids = self.ids_config

        net_eff = ids.get("network", {}).get("eff", 0.0)
        net_cycle = ids.get("network", {}).get("cycle", 100)
        detection_time = (
            self.base_attack_time
            if self.model_options.history_mode == "local_step"
            else self.total_attack_time
        )
        f_net = 1 - (1-net_eff) ** (detection_time / net_cycle)

        comp_eff = ids.get(self.target_component, {}).get("eff", 0.0)
        comp_cycle = ids.get(self.target_component, {}).get("cycle", 100)
        f_local = 1 - (1-comp_eff) ** (self.base_attack_time / comp_cycle)

        return 1 - (1 - f_net) * (1 - f_local)

    def compute_passive_success_probability(self):
        fw_net = self.firewall_config.get("network", 0.0)
        fw_local = self.firewall_config.get(self.target_component, 0.0)
        if isinstance(fw_net, dict):
            fw_net = fw_net["eff"]
        if isinstance(fw_local, dict):
            fw_local = fw_local["eff"]
        firewall_effect = (1 - fw_net) * (1 - fw_local)

        patch_eff = self.patch_config.get(self.vulnerability.name, 0.0)

        return (
            self.vulnerability.base_success
            * firewall_effect
            * (1 - patch_eff)
        )

    def _event_probabilities(self, p_response, has_response):
        """Probabilities conditional on this attack and the supplied response policy."""
        if isinstance(p_response, (bool, str)) or not np.isfinite(p_response) or not 0 <= p_response <= 1:
            raise ValueError("p_response must be finite and in [0, 1]")
        detection = float(np.clip(self.detected_prob, 0.0, 1.0))
        response = float(p_response) if has_response else 0.0
        return (
            1.0 - detection,
            detection * (1.0 - response),
            detection * response,
        )

    def compute_success_probability(self, p_response=0.0, *, has_response=True):
        """Success under the explicit conditional response probability (default: none)."""
        not_detected, detected_no_response, detected_response = (
            self._event_probabilities(p_response, has_response)
        )
        return (
            (not_detected + detected_no_response) * self.passive_success_prob
            + detected_response * self.active_success_prob
        )

    def get_utility_no_response(self):
        """Attacker utility when the defender does not respond."""
        return (
            self.passive_success_prob * self.attack_benefit
            - self.attack_cost
        )

    def get_utility_not_detected(self):
        return self.get_utility_no_response()

    @property
    def response_success_prob(self):
        return min(
            1.0,
            self.vulnerability.defense_success_prob
            * self.model_options.response_effectiveness_multiplier,
        )

    def get_utility_detected_no_response(self, exposure_penalty_ratio=None):
        if exposure_penalty_ratio is None:
            exposure_penalty_ratio = self.model_options.exposure_penalty_ratio
        return (
            self.passive_success_prob * self.attack_benefit
            - self.attack_cost
            - exposure_penalty_ratio * self.attack_cost
        )

    def get_utility_detected_with_response(self, exposure_penalty_ratio=None):
        if exposure_penalty_ratio is None:
            exposure_penalty_ratio = self.model_options.exposure_penalty_ratio
        return (
            self.active_success_prob * self.attack_benefit
            - self.attack_cost
            - exposure_penalty_ratio * self.attack_cost
        )

    def get_utility_with_response(self, penalty_ratio=None):
        """Attacker utility when detection triggers the response action."""
        return self.get_utility_detected_with_response(penalty_ratio)

    def get_defend_loss_no_response(self):
        """Defender loss when the defender does not respond."""
        return self.passive_success_prob * self.defend_loss

    def get_defend_loss_not_detected(self):
        return self.get_defend_loss_no_response()

    def get_defend_loss_detected_no_response(self):
        return self.get_defend_loss_no_response()

    def get_defend_loss_detected_with_response(self, response_operational_disruption_cost=0.0):
        return (
            self.active_success_prob * self.defend_loss
            + response_operational_disruption_cost
        )

    def get_defend_loss_with_response(self):
        """Defender loss after a successful detection and response."""
        return self.get_defend_loss_detected_with_response(
            self.response_operational_disruption_cost
        )

    def get_expected_utility(self, penalty_ratio=None, *, p_response=0.0, has_response=True):
        """Expected attacker payoff for this action, including detection-only exposure."""
        probabilities = self._event_probabilities(p_response, has_response)
        utilities = (
            self.get_utility_not_detected(),
            self.get_utility_detected_no_response(penalty_ratio),
            self.get_utility_detected_with_response(penalty_ratio),
        )
        return sum(probability * utility for probability, utility in zip(probabilities, utilities))

    def get_defend_expected_loss(self, p_response=0.0, *, has_response=True):
        """Expected defender loss for this action and the supplied response policy."""
        not_detected, detected_no_response, detected_response = (
            self._event_probabilities(p_response, has_response)
        )
        return (
            (not_detected + detected_no_response) * self.get_defend_loss_no_response()
            + detected_response * self.get_defend_loss_with_response()
        )

    def __repr__(self):
        return (
            f"[{self.vulnerability.name} @ {self.target_component}] "
            f"SuccessProb(r=0)={self.success_prob:.3f}, "
            f"DetectProb={self.detected_prob:.3f}, "
            f"TotalTime={self.total_attack_time:.2f}, "
            f"Utility(r=0)={self.get_expected_utility():.2f}, "
            f"DefendLoss(r=0)={self.get_defend_expected_loss():.2f}"
        )


class DoNothingStrategy:
    def __init__(self):
        self.success_prob = 0.0
        self.detected_prob = 0.0
        self.attack_cost = 0.0
        self.attack_benefit = 0.0
        self.defend_loss = 0.0
        self.base_attack_time = 0.0
        self.total_attack_time = 0.0

    def compute_success_probability(self, p_response=0.0, *, has_response=True) -> float:
        return 0.0

    def get_expected_utility(self, penalty_ratio=None, *, p_response=0.0, has_response=True) -> float:
        return 0.0

    def get_defend_expected_loss(self, p_response=0.0, *, has_response=True) -> float:
        return 0.0

    def __repr__(self):
        return "[Do Nothing]"
